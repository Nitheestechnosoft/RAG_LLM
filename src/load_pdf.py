from pypdf import PdfReader #importing the Pdfreader package


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text=""
    for page_number, page in enumerate(reader.pages): #enumerate provided the page number and content and is assigned to the page_number and page variables
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_number + 1} ---\n" #assigning the page number to the text and so making the chunking process easy
            full_text += text
        return full_text
