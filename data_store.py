
import fitz
from rich import print
from sentence_transformers import SentenceTransformer


def data_extract_chunk(file_path):
    doc=fitz.open(file_path)
    # print(doc.page_count)

    fitz.TOOLS.mupdf_display_errors(False)
    content=[]

    for page_number in range(len(doc)):
        page=doc.load_page(page_number)
        page_text=page.get_text("text")
        stripped = page_text.split()
        chunk_size=1000
        chunk_overload=100

        for j in range(0,len(stripped), chunk_size):
            chunk__text = " ".join(list(stripped[j:j+chunk_size]))

            content.append(
                {
                    "page_number":page_number + 1,
                    "text":chunk__text
                }
            )


    embedding=SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5",trust_remote_code=True)
    em=embedding.encode("hello")

    page_content=[chunk['text'] for chunk in content]
    page_number=[chunk['page_number'] for chunk in content]
    content_embedding=[embedding.encode(chunk['text']) for chunk in content]
    
    return content,content_embedding


# data_extract_chunk(file_path="D:\\FSS\\ai-test\\MACHINE LEARNING(R17A0534) (1).pdf")


def data_store_vectorstore(content,content_embedding):
        

def data_retrieve(query):


