# Variable Name Normalizer

A Streamlit application that uses Google's Gemini AI to normalize variable names based on standard conventions and examples.

## Features

- **Single Variable Normalization**: Normalize individual variable names in real-time
- **Batch Normalization**: Process multiple variables at once via manual input or CSV upload
- **Custom Mappings**: Add your own mappings for specific variable names
- **Example Browsing**: View and search through example mappings

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd variable-normalizer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API key:
   - Rename `.env.example` to `.env`
   - Add your Google API key to the `.env` file
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

## How it Works

The application uses a set of example variable name mappings stored in `data/filed-mappings.csv` to learn how to normalize new variable names. It uses the Gemini 1.5 Flash API to generate normalized names based on these examples.

Each normalization includes:
- A display name (with proper spacing and capitalization)
- A programming variable name (in snake_case)
- The likely category of the variable
- A confidence score
- An explanation of the normalization process

## Custom Mappings

You can add your own custom mappings using the sidebar. These mappings will take precedence over the AI's suggestions for specific variable names.

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Google Generative AI SDK
- An API key for Google's Gemini

## Limitations

- The service requires an internet connection to access the Gemini API
- The quality of normalizations depends on the examples provided
- API rate limits may apply depending on your Google API usage
