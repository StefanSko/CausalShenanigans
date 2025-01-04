import os
import re
from datetime import datetime
from bs4 import BeautifulSoup
import nbformat
from nbconvert import HTMLExporter


class BlogPostGenerator:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self.docs_path = os.path.join(repo_path, "docs")
        self.posts_path = os.path.join(self.docs_path, "posts")
        self.templates_path = os.path.join(self.docs_path, "templates")

        # Create necessary directories
        os.makedirs(self.posts_path, exist_ok=True)
        os.makedirs(self.templates_path, exist_ok=True)

    def _find_existing_post(self, slug):
        """Find an existing post with the given slug."""
        for filename in os.listdir(self.posts_path):
            if filename.endswith(f'-{slug}.html'):
                return os.path.join(self.posts_path, filename)
        return None

    def process_notebook(self, notebook_path, title=None, date=None):
        """Convert a Jupyter notebook to a blog post."""
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Extract title from first heading if not provided
        if not title:
            title = self._extract_title(nb)

        # Use current date if not provided
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')

        # Generate slug from title
        slug = self._generate_slug(title)

        # Check for existing post
        existing_post = self._find_existing_post(slug)
        is_replacement = False
        if existing_post:
            # Extract date from existing post filename
            existing_date = os.path.basename(existing_post).split('-')[0]
            date = existing_date  # Preserve the original date
            os.remove(existing_post)  # Remove the old post
            is_replacement = True
            print(f"Replacing existing post from {date}")

        # Convert notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'basic'
        html_body, _ = html_exporter.from_notebook_node(nb)

        # Read the template
        with open(os.path.join(self.templates_path, 'post_template.html'), 'r', encoding='utf-8') as f:
            template = f.read()

        # Replace placeholders in template
        post_html = template.replace('{{title}}', title)
        post_html = post_html.replace('{{date}}', date)
        post_html = post_html.replace('{{content}}', html_body)

        # Save the post
        output_path = os.path.join(self.posts_path, f"{date}-{slug}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(post_html)

        # Update index page with correct replacement flag
        self._update_index(title, date, slug, is_replacement=is_replacement)

        print(f"Created blog post: {output_path}")

    def _extract_title(self, notebook):
        """Extract title from the first heading in the notebook."""
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                # Look for the first heading
                lines = cell.source.split('\n')
                for line in lines:
                    if line.startswith('#'):
                        return line.lstrip('#').strip()

        # Fallback to notebook filename if no heading found
        return "Untitled Post"

    def _generate_slug(self, title):
        """Generate a URL-friendly slug from title."""
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug

    def _update_index(self, title, date, slug, is_replacement=False):
        """Update the index.html file with the new post."""
        index_path = os.path.join(self.docs_path, 'index.html')

        with open(index_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # If replacing, remove the old entry first
        if is_replacement:
            post_list = soup.find('ul', class_='post-list')
            existing_link = post_list.find('a', href=f"posts/{date}-{slug}.html")
            if existing_link:
                existing_link.find_parent('li').decompose()

        # Create new post entry
        post_list = soup.find('ul', class_='post-list')
        new_post = soup.new_tag('li', attrs={'class': 'post-item'})

        date_div = soup.new_tag('div', attrs={'class': 'post-date'})
        date_div.string = date

        link = soup.new_tag('a', href=f"posts/{date}-{slug}.html", attrs={'class': 'post-title'})
        link.string = title

        excerpt = soup.new_tag('div', attrs={'class': 'post-excerpt'})
        excerpt.string = f"A new blog post about {title.lower()}..."

        new_post.append(date_div)
        new_post.append(link)
        new_post.append(excerpt)

        # Add new post at the beginning of the list
        if post_list.li:
            post_list.li.insert_before(new_post)
        else:
            post_list.append(new_post)

        # Save updated index
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(str(soup.prettify()))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate blog posts from Jupyter notebooks')
    parser.add_argument('notebook', help='Path to the Jupyter notebook')
    parser.add_argument('--title', help='Blog post title (optional)')
    parser.add_argument('--date', help='Blog post date (YYYY-MM-DD)')

    args = parser.parse_args()


    blog = BlogPostGenerator()

    if args.notebook:
        blog.process_notebook(args.notebook, args.title, args.date)