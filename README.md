# AI Endpoint Boilerplate

Boilerplate app for fast AI endpoint creation from Hugging Face

## Prerequisites

Make sure you have Python version >=3.8.x installed.

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the API endpoint:
```bash
uvicorn app.main:app --reload
```

## Testing

- Access the Swagger UI documentation at `/docs`
- Use the interactive API documentation to test your endpoints