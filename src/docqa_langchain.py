from dataclasses import dataclass, asdict
from typing import Union, List, Optional
from pathlib import Path
import logging

import mlflow

from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(filename)s--l.%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

# TODO: 
# * add parameter for the splitter method to use
@dataclass
class DocQAParameters:
    """Input parameters for DocQA class"""
    nb_chunks_retrieved: int = 4

    embed_model_name: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 200
    add_start_index: bool = True


class DocQA:
    def __init__(self, parameters: DocQAParameters = DocQAParameters(), cache_folder: Optional[str]="experiments/model_cache/"):
         self.parameters = parameters
         
         self.document_source = None
         self.vector_db = None
         
         self.embed_model = HuggingFaceEmbeddings(model_name=self.parameters.embed_model_name, cache_folder=cache_folder)

    def query(self, question: str, document: Optional[Union[Document, str]]=None) -> List[Document]:
        """Ask a question about a document

        Args:
            document (Union[Document, str]): document to interrogate. If a `str`, it will be loaded
            question (str): A question about the text

        Returns:
            List[Document]: chunks from the input document that should help answer the input questions
        """
        if not (document is None or document is self.document_source):
            logger.debug("Found new document to embed")
            self.document_source = document
            self.vector_db = None
            # Load document
            if isinstance(document, str):
                logger.debug(f"Load document from path {document}")
                document = self.load_document(document)
            
            # Chunk and embed document
            logger.debug("Chunking and embedding document")
            self.embed_document(document=document)

        # Query documents
        if self.vector_db is None:
            raise ValueError("You need to populate your vector_db first")
        logger.debug(f"Retrieving chunks for question: {question}")
        return self.vector_db.similarity_search(question, k=self.parameters.nb_chunks_retrieved)
    
    @staticmethod
    def load_document(document_path: Path) -> List[Document]:
        """Load documents from a path

        Args:
            path (Path): path to the documents

        Returns:
            List[Document]: loaded documents
        """
        loader = TextLoader(document_path)
        document= loader.load()
        logger.debug(f"Loaded {len(document)} document(s) from path {document_path}")
        return document
    
    def embed_document(self, document: List[Document]) -> None:
        """Go from document to embedded chunks

        Args:
            document (List[Document]): documents to embed
        """
        chunks = self._chunk_document(document=document)
        self._embed_chunks(chunks=chunks)
    
    def _chunk_document(self, document: List[Document]) -> List[Document]:
        """Split the input document into chunks

        Args:
            document (List[Document]): text documents to chunk

        Returns:
            List[Document]: list of chunks coming from the input document
        """
        text_splitter = CharacterTextSplitter(
            chunk_size = self.parameters.chunk_size,
            chunk_overlap = self.parameters.chunk_overlap,
            add_start_index = self.parameters.add_start_index,
        )
        chunks = text_splitter.split_documents(document) # TODO: Except a list and gets a Document
        logger.debug(f"Split the document into {len(chunks)} chunks.")
        return chunks
    
    def _embed_chunks(self, chunks: List[Document]) -> None:
        """Embed each Document chunk

        Args:
            chunks (List[Document]): List of Document objects that correspond to the chunks
        """
        self.vector_db = FAISS.from_documents(chunks, self.embed_model)


if __name__ == "__main__":
    mlflow.set_experiment("TEST/Doc Q&A -- Langchain")
    with mlflow.start_run() as run:
        logger.info(f"MLFlow run id: {run.info.run_id}")
        docqa = DocQA()
        mlflow.log_params(asdict(docqa.parameters))
        document_path = "experiments/docs/state_of_the_union.txt"
        question="By how much will the deficit be down by the end of this year?"
        retrieved_docs = docqa.query(question=question, document=document_path)
        mlflow.log_params(
            {
                "document_path": document_path,
                "question": question,
                "retrieved_docs": retrieved_docs
            }
        )
        print(retrieved_docs[0])