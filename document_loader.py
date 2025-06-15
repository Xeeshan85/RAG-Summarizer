import os
import re
import PyPDF2
import markdown


class DocumentLoader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.md']
    
    def load_document(self, path: str) -> str:
        file_ext = os.path.splitext(path)[1]
        file_ext = file_ext.lower()
        
        if file_ext == '.pdf':
            return self._load_pdf(path)
        elif file_ext == '.txt':
            return self._load_txt(path)
        elif file_ext == '.md':
            return self._load_markdown(path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_pdf(self, path: str) -> str:
        text = ''
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
        return text
    
    def _load_txt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()

    def _load_markdown(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        html = markdown.markdown(content) ## convert to html
        text = re.sub('<[^<]+?>', '', html) ## remove html tags
        return text
