import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
import time
from streamlit_star_rating import st_star_rating
import subprocess
import os, uuid
import altair as alt
import sqlite3
import matplotlib.pyplot as plt
import threading
from captcha.image import ImageCaptcha
import random
import string
from src.classifier import Classifier
from pathlib import Path

# add project directory as a path for importing modules
#import sys
#sys.path.append('/project/kumar-lab/chouda/jabs/Webapp')

#from ClassifierOutputStatistics import plot_bout_dist

length_captcha = 4
width = 200
height = 150

# Set the page configuration
st.set_page_config(page_title="JABS", page_icon="🐁", layout='wide')

def get_table_download_link(df):
    """Function for handling file downloads"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="classifier_data.csv">Download CSV File</a>'
    return href

def dataframe_with_selections(df):
    """Function for creating the DataFrame with selection functionality"""
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    edited_df = st.data_editor(df_with_selections)
    selected_indices = list(np.where(edited_df.Select)[0])
    selected_rows = df[edited_df.Select]
    return {"selected_rows_indices": selected_indices, "selected_rows": selected_rows}


def create_unique_folder(user_email):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    unique_id = uuid.uuid4().hex  # generates a unique ID
    folder_name = f"submissions/{user_email}_{timestamp}_{unique_id}"

    # create the unique folder
    os.makedirs(folder_name, exist_ok=True)

    return folder_name


def sidebar_elements():
    """Function for adding elements to the sidebar"""
    st.sidebar.image('data/jax_logo.png', width=200)  # Replace 'logo.png' with the path to your logo image
    st.sidebar.title('JAX Animal Behavior System')
    
    st.sidebar.markdown("""
    [![Stars](https://img.shields.io/github/stars/KumarLabJax?style=social)](https://github.com/KumarLabJax)
    [![Twitter](https://img.shields.io/twitter/follow/vivekdna?style=social)](https://twitter.com/vivekdna)
    """)
    #st.sidebar.header("📤 Upload Classifier or Pose File", help="Should be JABS compatible pose file")
    st.sidebar.header("📤 Upload Classifier ", help="Should be JABS compatible classifier")
    #upload_type = st.sidebar.radio("", ('Classifier', 'Pose'),  horizontal=True)
    uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=False)

    # Handle file upload
    handle_file_upload(uploaded_files, behavior_name)

    behavior_name = st.sidebar.text_input("Behavior Name", help="Name of the behavior")

    #return upload_type, uploaded_files
    return uploaded_files, behavior_name


def run_script(env_vars, done_flag):
    """Function for running the script
    env_vars: dictionary of environment variables to pass to the script
    """
  
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Define log files
    #with open('logs/output.log', 'w') as stdout_file, open('logs/error.log', 'w') as stderr_file:
    #    # Run the script and redirect the output and errors to files
    #    process = subprocess.Popen(['infer.sh'], stdout=stdout_file, stderr=stderr_file, env=env_vars)

    #    # Wait for the process to complete
    #    process.wait()
    
    # Signal that the script is done
    done_flag['done'] = True


def captcha_control():
    # If 'captcha_passed' not defined, initialize it as False
    if 'captcha_passed' not in st.session_state:
        st.session_state['captcha_passed'] = False

    if st.session_state['captcha_passed'] == False:
        col1, col2 = st.sidebar.columns(2)
        
        if 'Captcha' not in st.session_state:
            st.session_state['Captcha'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length_captcha))
        
        image = ImageCaptcha(width=width, height=height)
        data = image.generate(st.session_state['Captcha'])
        col1.image(data)
        capta2_text = col2.text_input('Enter captcha text')
        
        if st.sidebar.button("Verify the code"):
            capta2_text = capta2_text.replace(" ", "")
            if st.session_state['Captcha'].lower() == capta2_text.lower().strip():
                st.session_state['captcha_passed'] = True
                st.experimental_rerun() 
            else:
                st.sidebar.error("Incorrect CAPTCHA code, please try again.")
                del st.session_state['Captcha']
                st.session_state['captcha_passed'] = False
                st.experimental_rerun()
        else:
            st.stop()

#def handle_file_upload(upload_type, uploaded_files):
def handle_file_upload(uploaded_files, behavior_name):
    """Function for handling file uploads"""

    #smtp_password = os.getenv('SMTP_PASSWORD')

    if uploaded_files is not None:

        #TODO: Add a check for the file type (should be a binary pickle file)

        # Add your Classifier file validation here
        # Validate if the uploaded classifier has window length of (2, 5, 10, 15, 20, 60 , If not, throw an error )
        classifier_file = os.path.basename(uploaded_files.name)
        print(uploaded_files)
        

        #try:
        #    classifier1 = Classifier()
        #    classifier1.load(uploaded_files)

        #except ValueError as e:
        #    st.error(f"Unable to load classifier from {classifier_file}:", icon="🚫")
        #    sys.exit(e)
    
        # if classifier has no attribute window_length, throw an error 
        classifier_flag = True
      
        #if not hasattr(classifier1, 'window_size'):
        #    st.error(f"Classifier file must have a window_length attribute.",  icon="🚨")
        #    st.info(f'Upload a valid classifier file.')
        #    classifier_flag = False

        #if classifier1.window_size not in [2, 10, 15, 20, 60]:
        #    st.error(f"Classifier window length must be one of [2, 5, 10, 15, 20, 60].",  icon="🚨")
        #    st.info(f'Upload a valid classifier file.')
        #    classifier_flag = False
        
        if classifier_flag:
            st.sidebar.text("File uploaded. Please select a dataset.")
            dataset_dir_name = st.sidebar.selectbox("Select Dataset", options=['JABS600', 'JABS1200', 'tiny'], help='JABS 600 has 598 1hr OFA videos consisting of 60 strains of mice, JABS1200 has 1139 1-hr OFA videos distributed over 60 strains of mice')
            email = st.sidebar.text_input("Enter your email")
            if 'captcha_passed' in st.session_state and st.session_state['captcha_passed']:

                # Check if 'submit_clicked' is in the session state, if not, initialize it
                if 'submit_clicked' not in st.session_state:
                    st.session_state['submit_clicked'] = False

                if not st.session_state['submit_clicked']:
                    if st.sidebar.button("Submit"):
                        # Set the done flag to False initially
                        done_flag = {'done': False}

                        folder_name = create_unique_folder(email) 
                
                        with open(f'{folder_name}/{uploaded_files.name}', 'wb') as f:
                            f.write(uploaded_files.getbuffer())
                        
                        classifier = os.path.basename(uploaded_files.name)              
                        env = os.environ.copy()
                        env["SUBMISSION_ID"] = folder_name
                        env["CLASSIFIER"] = classifier
                        env["BEHAVIOR_NAME"] = behavior_name
                        env["DATASET_DIR_NAME"] = dataset_dir_name 

                        # Start a separate thread to run the script
                        #thread = threading.Thread(target=run_script, args=(env, done_flag))
                        #thread.start()

                        st.sidebar.success(f"Submission for {email} received and results emailed!")
                        st.session_state['submit_clicked'] = True
                    
                else:
                    st.sidebar.markdown("*You have already submitted your selections*") 

def load_classifier_library():
    """Load the classifier library with average ratings from the database."""
    # Read the classifier library from CSV
    try:
        df = pd.read_csv("data/jabs_classifiers/classifier_info.csv")
    except FileNotFoundError:
        st.error("File not found. Please check the file path and try again.")

    # Connect to the SQLite database
    with sqlite3.connect('data/ratings.db') as conn:
        # Create a cursor for database operations
        cur = conn.cursor()

        # Iterate over each classifier in the DataFrame
        for i, row in df.iterrows():
            classifier_name = row['ClassifierName']

            # Fetch the average rating for this classifier from the database
            cur.execute('''
                SELECT AVG(rating)
                FROM user_ratings
                WHERE classifier_name = ?;
            ''', (classifier_name,))

            # Get the result of the query (average rating)
            avg_rating = cur.fetchone()[0]

            # If there are no ratings for this classifier, avg_rating will be None.
            # Replace it with some default value in that case (e.g. 0 or your previous default value).
            if avg_rating is None:
                avg_rating = 0

            # Update the rating in the DataFrame
            df.loc[i, 'Rating'] = avg_rating

    return df

def main():
    """Main function for the app"""

    # Load the classifier library
    df = load_classifier_library()
 
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
    try:
        df_heritability = pd.read_csv("data/genetic_correlations/PVE_GEMMA_estimates.txt")
    except FileNotFoundError:
        st.error("File not found. Please check the file path and try again.")

    if 'rating' not in st.session_state:
        st.session_state['rating'] = {}

    if 'df' not in st.session_state:
        st.session_state['df'] = df.copy()

    df = st.session_state['df']

    # Add elements to the sidebar
    #upload_type, uploaded_files = sidebar_elements()
    uploaded_files, behavior_name = sidebar_elements()
    print(uploaded_files)

    # Handle file uploads
    #handle_file_upload(upload_type, uploaded_files)
    

    # Use the function to add selection functionality to the DataFrame
    st.title("Classifier Library")
    selection = dataframe_with_selections(df)

    if len(selection["selected_rows"])>0:
        st.write("Your selection:")
        st.write(selection["selected_rows"])

    # Display the selected rows when more than one row is selected
    if len(selection["selected_rows_indices"]) ==2:
        if st.button('Get Genetic Correlations and Heritability Score'):
                time.sleep(2)
                selected_classifiers = selection["selected_rows"]["ClassifierID"]
                classifier_ids = [name for name in selected_classifiers]
                classifier_ids.sort()
                filename = f"data/genetic_correlations/gen_corr_{classifier_ids[0]}_{classifier_ids[1]}.csv"
                gen_corr_matrix = pd.read_csv(filename,index_col=0)
                gen_corr_melted = gen_corr_matrix.reset_index().melt(id_vars='index', var_name='phenotype1', value_name='value')
                gen_corr_melted = gen_corr_melted.rename(columns={"index": "phenotype2"})
                gen_corr_melted['value_abs'] = gen_corr_melted['value'].abs()

                heatmap = alt.Chart(gen_corr_melted).mark_square().encode(
                    x=alt.X('phenotype1:N', title=f'Phenotype ({classifier_ids[0]})'),
                    y=alt.Y('phenotype2:N', title=f'Phenotype ({classifier_ids[1]})'),
                    color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue')),
                    size='value_abs:Q'
                ).properties(width=500, height=400)

                heritability_scores_selected = df_heritability[df_heritability['ClassifierID'].isin(classifier_ids)]
                chart = alt.Chart(heritability_scores_selected).mark_bar().encode(
                    y=alt.Y('phenotype:N', title='Phenotype'),
                    x=alt.X('PVE:Q', title='Heritability Score'),
                    color='ClassifierID',
                    row='ClassifierID'
                ).properties(width=400, height=300)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.altair_chart(heatmap, use_container_width=False)
                
                with col2:
                    st.altair_chart(chart, use_container_width=False)

                st.success("Genetic correlations and heritability scores calculated!")

    if len(selection["selected_rows_indices"])==1:
        tabs = st.container()
        with tabs:
            t1, t2 = st.tabs([ '📈 Classifier Info', 'Rate Classifier'])
            with t1:
                st.write(selection["selected_rows"])
                file = "data/jabs_classifiers/" + selection["selected_rows"]['FileName'].values[0]
                with open(file, "rb") as fp:
                    st.download_button(
                            label="Download Classifier",
                            data=fp,
                            file_name=selection["selected_rows"]['FileName'].values[0],
                            mime='application/octet-stream'
                            )
            with t2:
                rating = st_star_rating("Please rate the classifier", maxValue=5, defaultValue=3, key="rating", emoticons=True)
                user_comment = st.text_input('Please leave your comments here:')
                user_email = st.text_input('Please enter your email here:')
                
                if st.button('Submit Rating'):
                    classifier_name = selection["selected_rows"]["ClassifierName"].values[0]
                    df.loc[df['ClassifierName'] == classifier_name, 'Rating'] = rating
                    st.session_state['df'] = df

                    # Connect to the SQLite database (will be created if it doesn't exist)
                    with sqlite3.connect('data/ratings.db') as conn:
                        # Create the table if it doesn't exist
                        conn.execute('''
                            CREATE TABLE IF NOT EXISTS user_ratings (
                                classifier_name TEXT,
                                user_email TEXT,
                                rating INT,
                                rating INT,
                                comment TEXT
                            );
                        ''')

                        # Insert the new rating
                        conn.execute('''
                            INSERT INTO user_ratings (classifier_name, user_email, rating, comment)
                            VALUES (?, ?, ?, ?);
                        ''', (classifier_name, user_email, rating, user_comment))

                        conn.commit()
                    st.success('Thanks for your rating!')

if __name__ == "__main__":
    main()
    captcha_control()



