import fitz  # PyMuPDF
import io

def test_highlight_pdf_text(pdf_path, highlight_text="• Name: Sarah Johnson"):
    try:
        # Open the PDF file
        document = fitz.open(pdf_path)

        # Iterate through all pages
        for page_num in range(len(document)):
            page = document.load_page(page_num)  # PyMuPDF uses 0-based indexing
            text_instances = page.search_for(highlight_text)

            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update()
                print(f"Highlighted '{highlight_text}' on page {page_num + 1}")
            else:
                print(f"No matching text found on page {page_num + 1} for '{highlight_text}'")

        # Save the modified PDF to a temporary in-memory buffer
        temp_file = io.BytesIO()
        document.save(temp_file)
        document.close()
        temp_file.seek(0)  # Ensure the stream is at the start

        return temp_file, None  # No error

    except Exception as e:
        print(f"Error in test_highlight_pdf_text: {e}")
        return None, e  # Return the error

# Example usage
pdf_path = "/Users/sarthakgarg/Documents/Sarthak/Medical Record 01.pdf"  # Replace with your PDF file path
highlight_text = "• Name: Sarah Johnson"  # The text we want to highlight

# Run the test function
highlighted_pdf, error = test_highlight_pdf_text(pdf_path, highlight_text)

if highlighted_pdf:
    print("Highlighting successful. You can now save the highlighted PDF.")
    # Save the in-memory PDF to a file
    output_path = "/Users/sarthakgarg/Downloads/highlighted_output24.pdf"
  # The desired output file path
    with open(output_path, "wb") as f:
        f.write(highlighted_pdf.getbuffer())
    print(f"Highlighted PDF saved as {output_path}")
else:
    print(f"Highlighting failed: {error}")
