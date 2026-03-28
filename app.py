from pdfreader import read_pdf
from chunker import chunk_pages
from embedding import embed_chunks
from vector_db import store_embeddings

pdf_path = r"C:\Users\Ganesha Janani\Desktop\NHPS-POLICIES-Revised-1.pdf"
def run():
    print("Create embeddings...")
    pages=read_pdf(pdf_path)
    # print("Pages count: ",len(pages))
    # print(pages[2])
    # # print(f"Total of {len(pages)} pages from the pdf")
    # # print(pages[10] if pages else "No content.")

    chunks=chunk_pages(pages,chunk_size=900,chunk_overlap=150)
    # # print(f"Total chunks created:{len(chunks)}")
    # # print("82nd chunk:")
    # # print(chunks[83])

    chunks_embedded=embed_chunks(chunks)
    # print(f"Total embedded chunks: {len(chunks_embedded)}")
    # print(f"First Embedded Chunk:{chunks_embedded[0]}")

    store_embeddings(chunks,chunks_embedded)
   
if __name__=="__main__":
    run()
