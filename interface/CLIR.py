import click
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Get absolute paths
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
api_dir = project_root / 'API'

# Add API directory to system path
sys.path.append(str(api_dir))

from indexerAPI import IndexerAPI
from retrievalAPI import RetrievalAPI
from wordPredictAPI import WordPredictAPI

# Initialize the APIs
indexer = IndexerAPI()
retriever = RetrievalAPI()
predictor = WordPredictAPI()

# Ensure data directories exist
notes_dir = current_dir/ 'data' / 'notes'
embeddings_dir = api_dir / 'data' / 'embeddings'

# Create directories if they don't exist
notes_dir.mkdir(parents=True, exist_ok=True)
embeddings_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ['NOTES_DIRECTORY'] = str(notes_dir)
os.environ['EMBEDDINGS_DIRECTORY'] = str(embeddings_dir)

@click.group()
def cli():
    """Command-line interface for managing code-mixed Telugu-English notes."""
    pass

@cli.command()
@click.argument('filename')
def create(filename):
    """Create a new note with FILENAME."""
    try:
        # Check if file already exists
        note_path = notes_dir / f"{filename}.txt"
        if note_path.exists():
            raise FileExistsError(f"Note '{filename}' already exists")
            
        indexer.createNote(filename)
        click.echo(click.style(f"✓ Note '{filename}' created successfully.", fg='green'))
        click.echo(click.style(f"Note location: {note_path}", fg='blue'))
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))

@cli.command()
@click.argument('filename')
@click.argument('text')
def edit(filename ,text):
    """Edit note FILENAME with TEXT content."""
    try:
        note_path = notes_dir / f"{filename}.txt"
        if not note_path.exists():
            raise FileNotFoundError(f"Note '{filename}' not found")
            
        if text is None:
            click.echo(click.style("No changes made.", fg='yellow'))
            return
            
        indexer.editNote(filename, text)
        click.echo(click.style(f"✓ Note '{filename}' updated successfully.", fg='green'))
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))

@cli.command()
@click.argument('filename')
def delete(filename):
    """Delete note FILENAME and its embeddings."""
    try:
        note_path = notes_dir / f"{filename}.txt"
        bert_emb_path = embeddings_dir / f"{filename}_bert.npy"
        ri_emb_path = embeddings_dir / f"{filename}_ri.npy"
        
        # Delete files if they exist
        files_deleted = False
        if note_path.exists():
            note_path.unlink()
            files_deleted = True
            
        if bert_emb_path.exists():
            bert_emb_path.unlink()
            files_deleted = True
            
        if ri_emb_path.exists():
            ri_emb_path.unlink()
            files_deleted = True
            
        if files_deleted:
            click.echo(click.style(f"✓ Note '{filename}' and its embeddings deleted successfully.", fg='green'))
        else:
            click.echo(click.style(f"Note '{filename}' not found.", fg='yellow'))
            
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))

@cli.command()
@click.argument('query_text')
@click.option('--top-k', '-k', default=3, help='Number of results to return')
def search(query_text, top_k):
    """Search notes using QUERY_TEXT."""
    try:
        results = retriever.find(query_text)
        
        if not results:
            click.echo(click.style("No matching documents found.", fg='yellow'))
            return

        click.echo(click.style(f"\nTop {top_k} results for query: ", fg='blue') + 
                  click.style(f'"{query_text}"', fg='cyan'))
        click.echo("=" * 80)

        for idx, result in enumerate(results[:top_k], 1):
            click.echo(click.style(f"\nResult {idx}", fg='blue'))
            click.echo(click.style("-" * 40, fg='blue'))
            
            click.echo(click.style("Document: ", fg='green') + 
                      click.style(result['note_id'], fg='white'))
            click.echo(click.style("Similarity: ", fg='green') + 
                      click.style(f"{result['similarity']:.4f}", fg='white'))
            
            click.echo(click.style("\nContent:", fg='green'))
            click.echo(result['content'].strip())
            click.echo("-" * 80)

    except Exception as e:
        click.echo(click.style(f"✗ Error during search: {str(e)}", fg='red'))



from datetime import datetime

@cli.command()
def list():
    """List all available notes."""
    try:
        # Ensure directory exists
        if not notes_dir.exists():
            click.echo(click.style(f"Creating notes directory at: {notes_dir}", fg='yellow'))
            notes_dir.mkdir(parents=True, exist_ok=True)
        
        # List all .txt files
        notes = [f for f in notes_dir.glob('*.txt')]
        
        if not notes:
            click.echo(click.style("No notes found.", fg='yellow'))
            return

        click.echo(click.style("\nAvailable Notes:", fg='blue'))
        click.echo("=" * 40)
        
        for idx, note_path in enumerate(sorted(notes), 1):
            # Get note information
            size = note_path.stat().st_size
            last_modified = datetime.fromtimestamp(note_path.stat().st_mtime)
            last_modified_str = last_modified.strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if embeddings exist
            bert_emb = embeddings_dir / f"{note_path.stem}_bert.npy"
            ri_emb = embeddings_dir / f"{note_path.stem}_ri.npy"
            has_embeddings = bert_emb.exists() and ri_emb.exists()
            
            # Format output
            click.echo(
                click.style(f"{idx}. ", fg='green') +
                click.style(note_path.stem, fg='white') +
                click.style(f" ({size:,} bytes)", fg='cyan') +
                click.style(" [indexed]" if has_embeddings else " [not indexed]", 
                          fg='blue' if has_embeddings else 'yellow') +
                click.style(f" - Last modified: {last_modified_str}", fg='white')
            )

    except Exception as e:
        click.echo(click.style(f"✗ Error listing notes: {str(e)}", fg='red'))
        click.echo(click.style(f"Notes directory: {notes_dir}", fg='yellow'))
@cli.command()
@click.argument('filename')
def show(filename):
    """Display the content of a specific note."""
    try:
        note_path = notes_dir / f"{filename}.txt"
        if not note_path.exists():
            raise FileNotFoundError(f"Note '{filename}' not found")

        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get note information
        size = note_path.stat().st_size
        last_modified = datetime.fromtimestamp(note_path.stat().st_mtime)
        last_modified_str = last_modified.strftime("%Y-%m-%d %H:%M:%S")

        click.echo(click.style(f"\nNote: {filename}", fg='blue'))
        click.echo(click.style(f"Size: {size:,} bytes", fg='cyan'))
        click.echo(click.style(f"Last modified: {last_modified_str}", fg='cyan'))
        click.echo("=" * 40)
        click.echo(content.strip())

    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))
@cli.command()
def debug():
    """Show debug information about directories and files."""
    click.echo(click.style("\nDebug Information:", fg='blue'))
    click.echo("=" * 40)
    
    # Show current directory
    click.echo(click.style("Current Directory: ", fg='green') + 
              click.style(str(current_dir), fg='white'))
    
    # Show API directory
    click.echo(click.style("API Directory: ", fg='green') + 
              click.style(str(api_dir), fg='white'))
    
    # Show notes directory
    click.echo(click.style("Notes Directory: ", fg='green') + 
              click.style(str(notes_dir), fg='white'))
    click.echo(click.style("Notes Directory Exists: ", fg='green') + 
              click.style(str(notes_dir.exists()), fg='white'))
    
    if notes_dir.exists():
        notes = list(notes_dir.glob('*.txt'))
        click.echo(click.style("\nFound Notes:", fg='blue'))
        for note in notes:
            click.echo(click.style(f"- {note.name}", fg='white'))
    
    # Show embeddings directory
    click.echo(click.style("\nEmbeddings Directory: ", fg='green') + 
              click.style(str(embeddings_dir), fg='white'))
    click.echo(click.style("Embeddings Directory Exists: ", fg='green') + 
              click.style(str(embeddings_dir.exists()), fg='white'))
    
    if embeddings_dir.exists():
        embeddings = list(embeddings_dir.glob('*.npy'))
        click.echo(click.style("\nFound Embeddings:", fg='blue'))
        for emb in embeddings:
            click.echo(click.style(f"- {emb.name}", fg='white'))

@cli.command()
def check_notes():
    """Debug command to check all files in notes directory."""
    try:
        click.echo(click.style("\nChecking notes directory...", fg='blue'))
        click.echo(f"Directory path: {notes_dir}")
        
        # List all files (not just .txt)
        all_files = list(notes_dir.iterdir())
        txt_files = list(notes_dir.glob('*.txt'))
        
        click.echo(click.style(f"\nTotal files found: {len(all_files)}", fg='blue'))
        click.echo(click.style(f"Text files found: {len(txt_files)}", fg='blue'))
        
        click.echo(click.style("\nAll files in directory:", fg='blue'))
        for file in sorted(all_files):
            click.echo(f"- {file.name}")
            
        click.echo(click.style("\nText files only:", fg='blue'))
        for file in sorted(txt_files):
            click.echo(f"- {file.name}")
            
    except Exception as e:
        click.echo(click.style(f"✗ Error checking notes: {str(e)}", fg='red'))


@cli.command()
@click.argument('context')
@click.option('--top-k', '-k', default=3, help='Number of predictions to return')
def predict(context, top_k):
    """Predict next words based on context."""
    try:
        predictor = WordPredictAPI()
        predictions = predictor.predict(context, top_k)
        
        if not predictions:
            click.echo(click.style("No predictions available.", fg='yellow'))
            return
            
        click.echo(click.style(f"\nNext word predictions for: ", fg='blue') + 
                  click.style(f'"{context}"', fg='cyan'))
        click.echo("=" * 40)
        
        for word, score in predictions:
            is_english = all(ord(c) < 128 for c in word)
            word_color = 'green' if is_english else 'magenta'
            
            click.echo(click.style(f"- {word}: ", fg=word_color) + 
                      click.style(f"{score:.4f}", fg='white'))
                      
    except Exception as e:
        click.echo(click.style(f"✗ Error during prediction: {str(e)}", fg='red'))



@cli.command()
def train_predictor():
    """Retrain the word prediction model."""
    try:
        click.echo(click.style("Training word prediction model...", fg='yellow'))
        predictor.train(notes_dir)
        click.echo(click.style("✓ Word prediction model trained successfully.", fg='green'))
    except Exception as e:
        click.echo(click.style(f"✗ Error during training: {str(e)}", fg='red'))



if __name__ == '__main__':
    cli()