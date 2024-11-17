from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain_community.llms import HuggingFacePipeline
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from pathlib import Path
import json
import numpy as np
from functools import lru_cache
import pickle
import warnings

# Rest of the code remains the same...

@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-l6-v2"
    qa_model: str = "Intel/dynamic_tinybert"
    device: str = "cpu"
    max_length: int = 512
    temperature: float = 0.7
    cache_dir: str = "./cache"
    similarity_top_k: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 150
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding_model": self.embedding_model,
            "qa_model": self.qa_model,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "cache_dir": self.cache_dir,
            "similarity_top_k": self.similarity_top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

@dataclass
class QAMetrics:
    """Metrics for tracking QA system performance."""
    total_questions: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    error_count: int = 0
    
    def update_response_time(self, response_time: float) -> None:
        """Update average response time with new data point."""
        self.avg_response_time = (
            (self.avg_response_time * self.total_questions + response_time) /
            (self.total_questions + 1)
        )
        self.total_questions += 1

class AdvancedQASystem:
    def __init__(
        self,
        dataset_name: str = "databricks/databricks-dolly-15k",
        content_column: str = "context",
        model_config: Optional[ModelConfig] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the Advanced Question Answering System.
        
        Args:
            dataset_name: HuggingFace dataset name
            content_column: Column containing text content
            model_config: Configuration for models
            enable_caching: Whether to enable caching
        """
        self.dataset_name = dataset_name
        self.content_column = content_column
        self.model_config = model_config or ModelConfig()
        self.enable_caching = enable_caching
        
        # Initialize components
        self._initialize_components()
        
        # Setup metrics
        self.metrics = QAMetrics()
        
        # Setup caching
        if enable_caching:
            self._setup_cache()
        
        # Setup logging with more detailed configuration
        self._setup_logging()

    def _initialize_components(self) -> None:
        """Initialize all component variables."""
        self.raw_documents = None
        self.processed_documents = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_pipeline = None
        self.retriever = None
        self.qa_chain = None

    def _setup_cache(self) -> None:
        """Setup caching directory and files."""
        cache_dir = Path(self.model_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = cache_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.model_config.to_dict(), f)

    def _setup_logging(self) -> None:
        """Setup detailed logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('qa_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_state(self, path: str = "qa_system_state.pkl") -> None:
        """
        Save system state to disk.
        
        Args:
            path: Path to save state
        """
        state = {
            'vectorstore': self.vectorstore,
            'metrics': self.metrics,
            'config': self.model_config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info(f"System state saved to {path}")

    def load_state(self, path: str = "qa_system_state.pkl") -> None:
        """
        Load system state from disk.
        
        Args:
            path: Path to load state from
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.vectorstore = state['vectorstore']
        self.metrics = state['metrics']
        self.model_config = state['config']
        self.logger.info(f"System state loaded from {path}")

    @lru_cache(maxsize=1000)
    def _cached_get_answer(self, question: str) -> str:
        """Cached version of get_answer."""
        self.metrics.cache_hits += 1
        return self._get_answer_impl(question)

    def _get_answer_impl(self, question: str) -> str:
        """Internal implementation of get_answer."""
        if not self.qa_chain:
            self.initialize()
        return self.qa_chain.run({"query": question})

    def get_answer(self, question: str, return_metadata: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Get answer for a question with optional metadata.
        
        Args:
            question: Question string
            return_metadata: Whether to return metadata
            
        Returns:
            Answer string or tuple of (answer, metadata)
        """
        start_time = time.time()
        try:
            if self.enable_caching:
                answer = self._cached_get_answer(question)
            else:
                answer = self._get_answer_impl(question)
            
            response_time = time.time() - start_time
            self.metrics.update_response_time(response_time)
            
            if return_metadata:
                metadata = {
                    'response_time': response_time,
                    'cache_hit': self.enable_caching,
                    'source_documents': self.get_relevant_documents(question)
                }
                return answer, metadata
            return answer
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"Error getting answer: {str(e)}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'total_questions': self.metrics.total_questions,
            'avg_response_time': self.metrics.avg_response_time,
            'cache_hits': self.metrics.cache_hits,
            'error_count': self.metrics.error_count
        }

    def batch_process_questions(self, questions: List[str]) -> List[str]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of answers
        """
        return [self.get_answer(q) for q in questions]

    def evaluate_performance(self, test_questions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Evaluate system performance on test set.
        
        Args:
            test_questions: List of test questions
            ground_truth: List of correct answers
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.batch_process_questions(test_questions)
        metrics = {
            'avg_response_time': self.metrics.avg_response_time,
            'error_rate': self.metrics.error_count / max(1, self.metrics.total_questions)
        }
        return metrics

    # Previous methods remain the same...
    # (load_dataset, process_documents, setup_embeddings, etc.)