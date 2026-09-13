import fitz

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    document.close()

    return pages

if __name__ == "__main__":
    pdf_path = "data/Smart Tax Computation System Report.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print("Number of pages:", len(pages))

    for page in pages[:2]:
        print("\nPAGE", page["page"])
        print(page["text"][:500])

