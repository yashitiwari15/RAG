import fitz  # PyMuPDF

def highlight_sentence(pdf_path, sentence, output_path):
    # Open the PDF
    document = fitz.open(pdf_path)
    
    # Loop through all the pages
    for page_num in range(len(document)):
        page = document[page_num]
        
        # Search for the sentence
        text_instances = page.search_for(sentence)
        
        # If sentence is found, highlight it
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()
            print("Pdf highlighted")
    
    # Save the modified PDF
    document.save(output_path, garbage=4, deflate=True)
    document.close()

# Example usage
pdf_path = "/Users/sarthakgarg/Documents/Sarthak/Medical Record 01.pdf"
sentence = "• Name: Sarah Johnson"
output_path = "testfile.pdf"

highlight_sentence(pdf_path, sentence, output_path)
