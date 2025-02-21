# Proofreader Project

This project uses the Mistral 7B model to proofread, edit, and tidy up text files. It produces a new, nicely formatted markdown file with proper headings and a table of contents with links to the headings.

## Project Structure

```
proofreader
├── src
│   └── proofreader.py
├── texts
│   └── (your text files)
├── requirements.txt
└── README.md
```

## Getting Started

To run the script, follow these steps:

1. Ensure you have Python installed on your machine.
2. Install the required dependencies using the following command:

   ```
   pip install -r requirements.txt
   ```

3. Navigate to the project directory.
4. Run the script using the following command:

   ```
   python src/proofreader.py
   ```

## Dependencies

This project requires the following external dependencies:

- `transformers`
- `torch`
- `markdown`

These dependencies are listed in the `requirements.txt` file and can be installed using the command provided in the "Getting Started" section.