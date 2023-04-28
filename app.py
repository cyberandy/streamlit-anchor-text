import streamlit as st
from langchain import LLMChain, HuggingFaceHub, PromptTemplate

# Set up Streamlit
st.set_page_config(page_title="Anchor Text Optimization in SEO",
                   page_icon="img/fav-ico.png")

# Load Hugging Face API token from secrets.toml
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Get user input

# ---------------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------------- #
st.sidebar.image("img/logo-wordlift.png")
st.sidebar.title('Optimize your anchor text 🔎')
st.sidebar.header("Settings")
prompt_goes_here = st.sidebar.text_area("Enter your prompt:", value='''You are a SEO for a premium food and lifestyle brand.
    As content editor and SEO, read the keyword below along with the page's title and write an adequate and concise anchor text to reinforce the keyword targeting.
    Remain neutral and follow the examples below:
    Title: Affogato
    Keyword: affogato recipe
    Anchor text: affogato recipe

    Title: Secret Blueberry Muffins
    Keyword: bluberry muffins
    Anchor text: bluberry muffins''', height=300)
max_chars = st.sidebar.number_input(
    "Enter maximum number of characters for anchor text:", value=19)

st.sidebar.write("""
Try our latest tool to optimize your anchor text using LLMs.
\n\n
Have a question? [Let's talk](https://wordlift.io/contact-us) about it!.
\n\n
This is a SEO experiment by [WordLift](https://wordlift.io/).""")

# ---------------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------------- #

# Create form for title and main queries
form = st.form(key="form")
form.header("Title and Main Queries")
title_main_queries = form.text_area(
    "Enter title and main query pairs (one per line, comma separated):", value="Spicy stuffed sausage and cheese croissants, sausage croissant")
submit_button = form.form_submit_button(label="Submit")


def generate_anchor_text(_keyword, _title, max_chars=19):

    # Setting up the prompt
    prompt_tmp = prompt_goes_here + '''Title: {title}\n
    Keyword: {keyword}\n
    Anchor text:'''
    # Creating the prompt
    prompt = PromptTemplate(template=prompt_tmp,
                            input_variables=["keyword", "title"])
    run_statement = {"keyword": _keyword, "title": _title}

    # using HF
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(
        repo_id="google/flan-ul2", model_kwargs={"temperature": 0.4, "max_length": 64}))

    anchor_text = llm_chain.run(**run_statement)

    # Adding controls
    # Split the anchor_text into a list of words
    words = anchor_text.split()
    # Capitalize the first letter of each word
    words = [word.capitalize() for word in words]
    # Join the words back into a single string
    anchor_text = ' '.join(words)
    # While the length of anchor_text is greater than max_chars characters
    while len(anchor_text) > max_chars:
        # Remove the last word from the list of words
        words = words[:-1]
        # Join the remaining words back into a single string
        anchor_text = ' '.join(words)
        # Remove the last word from the list of words if it ends with "and", "&", or "for"
        last_word = words[-1]
        if last_word.lower() in ["and", "&", "for"]:
            words = words[:-1]
            anchor_text = ' '.join(words)

    return anchor_text


# Process form input
if submit_button:
    if not title_main_queries:
        st.error("Please enter at least one title and main query pair.")
    else:
        # Create table to display results
        title_main_query_pairs = title_main_queries.split("\n")

        # Create table to display results
        st.write("## Results")

        table_data = [["Title", "Main Query", "Anchor Text"]]

        for title_main_query in title_main_query_pairs:
            title, main_query = title_main_query.split(",")

            # Generate anchor text using provided function
            anchor_text = generate_anchor_text(
                main_query, title, max_chars=int(max_chars))

            # Add row to table data
            table_data.append([title, main_query, anchor_text])

        st.table(table_data)
