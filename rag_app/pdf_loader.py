from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path: str):
    loader = PdfReader(file_path)
    text = ""
    for page in loader.pages:
        text += page.extract_text() or ""
    return text
