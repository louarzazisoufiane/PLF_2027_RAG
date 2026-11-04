import os
from dotenv import load_dotenv

from langchain_docling import DoclingLoader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_cohere import CohereEmbeddings

load_dotenv()

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

custom_converter = DocumentConverter(
    format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
)

loader = DoclingLoader(
    file_path="./Docs", 
    converter=custom_converter 
)

docs = loader.load()


embeddings_model = CohereEmbeddings(
    model="cohere/embed-5-fast",
    cohere_api_key="COHERE_KEY"
)

