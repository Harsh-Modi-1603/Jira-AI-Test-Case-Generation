

import csv
import markdown
from markdown.treeprocessors import Treeprocessor
from markdown.extensions import Extension
from bs4 import BeautifulSoup

class TableExtractor(Treeprocessor):
    """A custom Treeprocessor that extracts tables from the HTML tree generated from markdown."""
    
    def run(self, doc):
        self.tables = []
        for element in doc.iter('table'):
            table_data = []
            for row in element.iter('tr'):
                row_data = [cell.text.strip() for cell in row.iter('td')]
                if row_data:
                    table_data.append(row_data)
            self.tables.append(table_data)
        return self.tables

class TableExtractorExtension(Extension):
    """Markdown extension to extract tables."""
    
    def extendMarkdown(self, md):
        md.treeprocessors.register(TableExtractor(md), 'tableextractor', 25)

def markdown_to_html(markdown_text):
    """Converts markdown text to HTML."""
    md = markdown.Markdown(extensions=[TableExtractorExtension()])
    return md.convert(markdown_text)

def extract_table_data(markdown_filename):
    """Extract table data from markdown file and return it."""
    with open(markdown_filename, 'r') as f:
        markdown_text = f.read()

    # Convert markdown to HTML
    html_content = markdown_to_html(markdown_text)
    
    # Parse HTML with BeautifulSoup to extract tables
    soup = BeautifulSoup(html_content, 'html.parser')
    
    tables = soup.find_all('table')
    
    table_data = []
    for table in tables:
        rows = table.find_all('tr')
        table_rows = []
        for row in rows:
            cols = row.find_all(['td', 'th'])  # include 'th' for headers
            cols = [ele.text.strip() for ele in cols]
            table_rows.append(cols)
        table_data.append(table_rows)
    
    return table_data

def save_to_csv(table_data, csv_filename):
    """Save extracted table data to a CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for table in table_data:
            for row in table:
                writer.writerow(row)
    print(f"CSV file saved as {csv_filename}")

def convert_markdown_to_csv(markdown_filename, csv_filename):
    """Convert markdown file to CSV."""
    table_data = extract_table_data(markdown_filename)
    save_to_csv(table_data, csv_filename)

# Example usage
convert_markdown_to_csv('test.md', 'test.csv')