import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please add your API key to the .env file.")
    st.stop()

# Initialize the Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Variable Name Normalizer",
    page_icon="🔄",
    layout="wide",
)

# Application title
st.title("🔄 Variable Name Normalizer")
st.markdown("""
This app helps normalize variable names using LLM. Upload your variable names 
or enter them manually, and get standardized variable names based on common conventions.
""")

# File paths
MAPPINGS_FILE = "data/filed-mappings.csv"

# Load example mappings
@st.cache_data
def load_example_mappings():
    try:
        df = pd.read_csv(MAPPINGS_FILE)
        return df
    except Exception as e:
        st.error(f"Error loading example mappings: {e}")
        return pd.DataFrame(columns=["Source Field", "Target Field", "Category"])

# Save custom mappings to CSV
def save_custom_mappings_to_csv(custom_mappings):
    try:
        # Load existing mappings
        df = pd.read_csv(MAPPINGS_FILE)
        
        # Create new rows for custom mappings
        new_rows = []
        for source, target in custom_mappings.items():
            # Check if this source field already exists
            if not df[df["Source Field"] == source].empty:
                # Update existing mapping
                df.loc[df["Source Field"] == source, "Target Field"] = target
            else:
                # Add new mapping
                new_rows.append({
                    "Source Field": source,
                    "Target Field": target,
                    "Category": "Custom Mappings"
                })
        
        # Append new rows if any
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(MAPPINGS_FILE, index=False)
        
        # Return success
        return True
    except Exception as e:
        st.error(f"Error saving custom mappings: {e}")
        return False

# Load mappings with cache invalidation option
@st.cache_data
def get_mappings(_reload=False):
    return load_example_mappings()

# Initialize or reload mappings
if 'mappings_loaded' not in st.session_state:
    st.session_state.mappings_loaded = True
    example_mappings = get_mappings()
elif st.session_state.get('reload_mappings', False):
    st.session_state.reload_mappings = False
    # Force reload by changing the parameter
    example_mappings = get_mappings(_reload=True)
    st.experimental_rerun()
else:
    example_mappings = get_mappings()

# Create the prompt for Gemini
def create_normalization_prompt(variable_name, examples, custom_mappings=None):
    # Check if the variable is already in custom mappings
    if custom_mappings and variable_name in custom_mappings:
        return None, custom_mappings[variable_name]
    
    # Check if the variable is already in the examples
    source_match = examples[examples["Source Field"] == variable_name]
    if not source_match.empty:
        return None, source_match.iloc[0]["Target Field"]
    
    # Get list of valid target fields to check against
    valid_targets = examples["Target Field"].unique().tolist()
    
    # Convert examples to text format
    examples_text = "\n".join([
        f"Source: {row['Source Field']}, Target: {row['Target Field']}, Category: {row['Category']}"
        for _, row in examples.head(50).iterrows()  # Limit to 50 examples
    ])
    
    # Include custom mappings if provided
    custom_mappings_text = ""
    if custom_mappings and len(custom_mappings) > 0:
        custom_mappings_text = "\nAdditionally, use these specific custom mappings:\n"
        custom_mappings_text += "\n".join([
            f"Source: {source}, Target: {target}"
            for source, target in custom_mappings.items()
        ])
    
    prompt = f"""
You are a variable name normalizer that follows standard naming conventions.
Given a variable name, convert it to a standardized form based on the examples below.

Examples of source variables and their normalized forms:
{examples_text}
{custom_mappings_text}

For the variable: "{variable_name}"

Normalize it following these rules:
1. Use spaces between words for display names (like "First Name")
2. Use snake_case for programming variables (like "first_name")
3. Expand abbreviations to their full form
4. Remove underscores, special characters, and standardize casing
5. If you recognize the category of the variable, mention it
6. IMPORTANT: Only normalize to one of these valid target fields: {', '.join(valid_targets)}
7. If the variable doesn't match any of the valid target fields, set display_name to "NA"

Return your answer in this JSON format:
{{
    "display_name": "The normalized display name with proper spacing and capitalization",
    "variable_name": "the_normalized_variable_name_in_snake_case",
    "category": "The likely category of this variable",
    "confidence": "High, Medium, or Low",
    "explanation": "Brief explanation of your normalization"
}}

If you cannot normalize the variable name with confidence or if it doesn't match any valid target field, set display_name to "NA" and category to "NA".
"""
    return prompt, None

# Function to call Gemini API
def normalize_variable(variable_name, custom_mappings=None):
    try:
        # Create prompt or get direct mapping
        prompt_text, direct_mapping = create_normalization_prompt(variable_name, example_mappings, custom_mappings)
        
        # If we have a direct mapping, return it without calling the API
        if direct_mapping:
            # Find the category from examples
            category = "Custom Mappings"
            for _, row in example_mappings.iterrows():
                if row["Target Field"] == direct_mapping:
                    category = row["Category"]
                    break
            
            return {
                "display_name": direct_mapping,
                "variable_name": direct_mapping.lower().replace(" ", "_"),
                "category": category,
                "confidence": "High",
                "explanation": "Direct mapping from existing examples or custom mappings"
            }
        
        # Create content for the model
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text)
                ]
            )
        ]
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
            response_mime_type="application/json",
        )
        
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config,
        )
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            response_text = response.text
            # Look for JSON content between curly braces if needed
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_content = response_text[json_start:json_end]
                result = json.loads(json_content)
            else:
                result = json.loads(response_text)
            
            # Set display_name to NA if confidence is Low
            if result.get("confidence", "").lower() == "low":
                result["display_name"] = "NA"
                result["category"] = "NA"
            
            # Verify the normalized name is in the valid targets
            valid_targets = example_mappings["Target Field"].unique().tolist()
            if result["display_name"] != "NA" and result["display_name"] not in valid_targets:
                result["display_name"] = "NA"
                result["category"] = "NA"
                result["explanation"] = "Normalized name not in list of valid target fields"
            
            return result
        except json.JSONDecodeError:
            st.error(f"Failed to parse response as JSON: {response_text}")
            return {
                "display_name": "NA",
                "variable_name": variable_name.lower().replace(" ", "_"),
                "category": "NA",
                "confidence": "Low",
                "explanation": "Failed to parse the model response"
            }
            
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None

# Initialize session state for custom mappings
if 'custom_mappings' not in st.session_state:
    st.session_state.custom_mappings = {}

# Sidebar - Custom Mappings
st.sidebar.header("Custom Mappings")
st.sidebar.write("Add your own mappings for specific variable names:")

# Input for adding custom mappings
with st.sidebar.form("custom_mapping_form"):
    col1, col2 = st.columns(2)
    with col1:
        source_field = st.text_input("Source Field", key="source_field")
    with col2:
        target_field = st.text_input("Target Field", key="target_field")
    
    save_to_csv = st.checkbox("Save to CSV file", value=False, help="Save this mapping to the filed-mappings.csv file for future use")
    
    submit_button = st.form_submit_button("Add Custom Mapping")
    
    if submit_button and source_field and target_field:
        st.session_state.custom_mappings[source_field] = target_field
        
        # Save to CSV if requested
        if save_to_csv:
            if save_custom_mappings_to_csv({source_field: target_field}):
                st.session_state.reload_mappings = True
        
        st.rerun()

# Display current custom mappings
if st.session_state.custom_mappings:
    st.sidebar.subheader("Your Custom Mappings")
    
    # Add a button to save all mappings to CSV
    if st.sidebar.button("Save All Custom Mappings to CSV"):
        if save_custom_mappings_to_csv(st.session_state.custom_mappings):
            st.sidebar.success("All custom mappings saved to CSV file!")
            st.session_state.reload_mappings = True
            st.rerun()
    
    for source, target in st.session_state.custom_mappings.items():
        cols = st.sidebar.columns([3, 1, 3, 1])
        cols[0].write(source)
        cols[1].write("→")
        cols[2].write(target)
        if cols[3].button("🗑️", key=f"delete_{source}"):
            del st.session_state.custom_mappings[source]
            st.rerun()
else:
    st.sidebar.info("No custom mappings added yet.")

# Main section
tab1, tab2, tab3 = st.tabs(["Normalize Single Variable", "Batch Normalization", "Example Mappings"])

# Tab 1: Single Variable Normalization
with tab1:
    st.subheader("Normalize a Single Variable")
    
    variable_name = st.text_input("Enter variable name to normalize:", placeholder="e.g. Cust_ID, F_Name, AcctNum")
    
    if st.button("Normalize", key="normalize_single") and variable_name:
        with st.spinner("Normalizing..."):
            result = normalize_variable(variable_name, st.session_state.custom_mappings)
            
            if result:
                st.success("Normalized Variable")
                st.info(f"Display Name: **{result['display_name']}**", icon="✅")
                st.info(f"Category: **{result['category']}**", icon="📁")
                st.info(f"Confidence: **{result['confidence']}**", icon="🎯")
                
                if result['explanation'] and result['display_name'] != "NA":
                    st.info(f"Explanation: {result['explanation']}", icon="💡")

# Tab 2: Batch Normalization
with tab2:
    st.subheader("Normalize Multiple Variables")
    
    # Option to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with variable names (single column)", type=["csv", "txt"])
    
    # Option to enter multiple variables manually
    manual_input = st.text_area(
        "Or enter variable names manually (one per line)",
        placeholder="F_Name\nLName\nCust_ID\nAcctNum"
    )
    
    if st.button("Normalize Batch", key="normalize_batch"):
        variables_to_normalize = []
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                if df.shape[1] == 1:
                    variables_to_normalize = df[0].tolist()
                else:
                    st.error("Please upload a CSV with a single column of variable names.")
            except Exception as e:
                st.error(f"Error processing upload: {e}")
        
        elif manual_input:
            variables_to_normalize = [v.strip() for v in manual_input.split("\n") if v.strip()]
        
        if variables_to_normalize:
            results = []
            progress_bar = st.progress(0)
            
            for i, var_name in enumerate(variables_to_normalize):
                with st.spinner(f"Processing {i+1}/{len(variables_to_normalize)}: {var_name}"):
                    result = normalize_variable(var_name, st.session_state.custom_mappings)
                    if result:
                        results.append({
                            "Original": var_name,
                            "Display Name": result["display_name"],
                            "Category": result["category"],
                            "Confidence": result["confidence"]
                        })
                progress_bar.progress((i + 1) / len(variables_to_normalize))
            
            if results:
                st.success(f"Processed {len(results)} variables")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="normalized_variables.csv",
                    mime="text/csv"
                )

# Tab 3: Example Mappings
with tab3:
    st.subheader("Example Mappings from Dataset")
    
    if not example_mappings.empty:
        # Filter options
        categories = ["All"] + sorted(example_mappings["Category"].unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories)
        
        # Search box
        search_term = st.text_input("Search mappings", "")
        
        filtered_df = example_mappings
        
        # Apply category filter
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["Category"] == selected_category]
        
        # Apply search filter
        if search_term:
            filtered_df = filtered_df[
                filtered_df["Source Field"].str.contains(search_term, case=False) |
                filtered_df["Target Field"].str.contains(search_term, case=False)
            ]
        
        # Display filtered examples
        st.dataframe(filtered_df, use_container_width=True)
        
        # Stats
        st.info(f"Showing {len(filtered_df)} of {len(example_mappings)} total mappings")
    else:
        st.error("No example mappings found. Please check the data/filed-mappings.csv file.")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit and Gemini-AI LLM. The normalization is based on example mappings "
    "and prompting. You can also add custom mappings in the sidebar to improve results."
) 