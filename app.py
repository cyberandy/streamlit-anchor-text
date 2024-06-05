import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import os
import json

# Set up Streamlit
st.set_page_config(
    page_title="Anchor Text Optimization in SEO", page_icon="img/fav-ico.png"
)

# Load Hugging Face API token from secrets.toml
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Get user input

# ---------------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------------- #
st.sidebar.image("img/logo-wordlift.png")
st.sidebar.title("Optimize your anchor text 🔎")
st.sidebar.header("Settings")
prompt_goes_here = st.sidebar.text_area(
    "Enter your prompt:",
    value="""You are a SEO for a premium food and lifestyle brand.
    As content editor and SEO, read the keyword below along with the page's title and write an adequate and concise anchor text to reinforce the keyword targeting.
    Remain neutral and follow the examples below:
    Title: Affogato
    Keyword: affogato recipe
    Anchor text: affogato recipe

    Title: Secret Blueberry Muffins
    Keyword: bluberry muffins
    Anchor text: bluberry muffins""",
    height=300,
)
max_chars = st.sidebar.number_input(
    "Enter maximum number of characters for anchor text:", value=19
)

repo_id = st.sidebar.selectbox(
    "Select your model",
    (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-v0.1",
        "tiiuae/falcon-7b-instruct",
        "bigscience/bloom",
        "google/flan-t5-xxl",
        "google/flan-ul2",
    ),
)
st.sidebar.write(
    """
Try our latest tool to optimize your anchor text using LLMs.
\n\n
This is a SEO experiment by [WordLift](https://wordlift.io/) on Generative AI applied to [Dynamic Internal Links](https://wordlift.io/blog/en/dynamic-internal-links-in-seo/).
\n\n
Have a question? [Let's talk](https://wordlift.io/contact-us) about it!."""
)

# ---------------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------------- #

# Create form for title and main queries
form = st.form(key="form")
form.header("Title and Main Query")
title_main_queries = form.text_area(
    "Enter title and main query pairs (one per line, comma separated):",
    value="Spicy stuffed sausage and cheese croissants, sausage croissant",
)
submit_button = form.form_submit_button(label="Submit")


def generate_anchor_text(_keyword, _title, _max_chars="19"):

    # Setting up the prompt
    prompt = f'''Glasses.com is a renowned retailer specializing in glasses and sunglasses online. As an SEO and content editor for Glasses.com, your task is to create a concise and appropriate anchor text to enhance keyword targeting, using the provided keyword and page title. Ensure to maintain a neutral tone and adhere to the examples below for guidance:

- Title: Arnette Glasses and Prescription Sunglasses
  - Keyword: arnette
  - Anchor text: Shop Arnette Glasses

- Title: Armani Exchange Glasses & Sunglasses: Prescription
  - Keyword: prescription sunglass man
  - Anchor text: A|X Prescription Sunglasses

- Title: Oakley® Prescription Sunglasses & Glasses
  - Keyword: sunglass
  - Anchor text: Browse Oakley Sunglasses

- Title: Starck Sunglasses & Glasses - Quality Eyewear
  - Keyword: starck biotech paris
  - Anchor text: Starck Biotech Paris Eyewear

- Title: Prescription Sunglasses for Men
  - Keyword: prescription sunglass man
  - Anchor text: Men's Prescription Sunglasses

- Title: Durable Prescription Eyeglasses for Women
  - Keyword: glass
  - Anchor text: Women's Eyeglasses

- Title: High-Quality Prescription Night Driving Glasses
  - Keyword: driving glasses
  - Anchor text: Night Driving Glasses

- Title: Stylish Prescription Sunglasses For Women
  - Keyword: prescription sunglasses for women
  - Anchor text: Women's Prescription Sunglasses

- Title: 30% off
  - Keyword: 30% off
  - Anchor text: Buy 30% off

- Title: Gucci Glasses
  - Keyword: Gucci Gc002013
  - Anchor text: Gucci Glasses

- Title: Arnette Glasses and Prescription Sunglasses
  - Keyword: arnette glasses
  - Anchor text: Arnette Glasses

- Title: Top Sunglasses Designer
  - Keyword: luxury sunglasses brands
  - Anchor text: Designer Sunglasses

- Title: Vogue Eyewear Sunglasses for Men
  - Keyword: Vogue Eyewear Sunglasses for Men
  - Anchor text: Vogue Eyewear Men

- Title: Saint Laurent Sunglasses for Women
  - Keyword: Saint Laurent Sunglasses for Women
  - Anchor text: SL Sunglasses Women

- Title: Burberry Kid's
  - Keyword: kids sunglasses
  - Anchor text: Burberry Kids

Now, based on these examples, review the title and the keyword below and provide in the response ONLY the Anchor text.

- Title: {_title}
  - Keyword: {_keyword}'''

    try:
        # Setting up the client for Hugging Face models
        client = InferenceClient(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

        # Correctly format the request according to InferenceClient expectations
        response = client.post(
            json={
                "inputs": prompt,
            },
            model="mistralai/Mistral-7B-Instruct-v0.3"
        )

        response = response.decode('utf-8')
        results = json.loads(response)

        # Extract the anchor text from the response
        anchor_text = results[0]['generated_text'].split('Anchor text:')[-1].strip()

        # Process the anchor text to fit the constraints
        words = anchor_text.split()
        words = [word.capitalize() for word in words]
        anchor_text = ' '.join(words)
        while len(anchor_text) > _max_chars:
            words = words[:-1]
            anchor_text = ' '.join(words)
            last_word = words[-1]
            if last_word.lower() in ["and", "&", "for"]:
                words = words[:-1]
                anchor_text = ' '.join(words)

        return anchor_text
    except Exception as e:
        print(f"An error occurred while generating anchor text: {str(e)}")
        return ""


# Process form input
if submit_button:
    if not title_main_queries:
        st.error("Please enter at least one title and main query pair.")
    else:
        # Create table to display results
        title_main_query_pairs = title_main_queries.split("\n")

        # Create table to display results
        st.write("## Results")

        table_data = []

        for title_main_query in title_main_query_pairs:
            title, main_query = title_main_query.split(",")

            # Generate anchor text using provided function
            anchor_text = generate_anchor_text(
                main_query, title, max_chars=int(max_chars)
            )

            # Add row to table data
            table_data.append([title, main_query, anchor_text])

        # Create dataframe from table data
        df = pd.DataFrame(table_data, columns=["Title", "Main Query", "Anchor Text"])

        st.table(df)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df(df)

        st.download_button(
            "Press to Download", csv, "file.csv", "text/csv", key="download-csv"
        )


# ---------------------------------------------------------------------------- #
# Adding the download button for the CSV
# ---------------------------------------------------------------------------- #

# Add a button to download the data
