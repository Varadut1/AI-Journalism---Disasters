import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from main import *
import csv
import random
import boto3
from lang import language_codes, language_codes_polly, merged_languages_with_voices, languages


s3 = boto3.client('s3', aws_access_key_id="AKIA4MTWKZ5UYUQDRGC5", aws_secret_access_key="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", region_name="ap-south-1")
translate = boto3.client('translate', aws_access_key_id="AKIA4MTWKZ5UYUQDRGC5", aws_secret_access_key="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", region_name="ap-south-1")

response = s3.list_objects_v2(Bucket="ccprojectnc", Prefix="Uploads" + '/')
if 'Contents' in response:
    # Folder already exists, find the highest numbered folder and increment it
    existing_folders = response['Contents'][-1]['Key'].split("/")[-2]
    new_folder_number = int(existing_folders)+ 1
    new_folder_name = f"Uploads/{new_folder_number}/"
    print(new_folder_name)
else:
    # Folder doesn't exist, create folder 1
    new_folder_name = f"Uploads/1/"

polly = boto3.client("polly", aws_access_key_id="AKIA4MTWKZ5UYUQDRGC5", aws_secret_access_key="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", region_name="ap-south-1")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.markdown("<h1 style='text-align: center; color: red;'>SKY NEWS</a></h1>", unsafe_allow_html = True)



def read_data_from_csv(file_path):
    labels_and_reports = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if it exists

        for row in reader:
            label = row[0].strip()
            report = row[1].strip()

            if label in labels_and_reports:
                labels_and_reports[label].append(report)
            else:
                labels_and_reports[label] = [report]

    return labels_and_reports

def generate_combined_report_from_csv(file_path, input_labels):
    labels_and_reports = read_data_from_csv(file_path)
    combined_report = ""

    for label in input_labels:
        if label in labels_and_reports:
            reports_for_label = labels_and_reports[label]
            selected_report = random.choice(reports_for_label)
            combined_report += selected_report + "\n"
        else:
            combined_report += f"No reports available for label '{label}'\n"

    return combined_report


# print(f"Report:\n{output_combined_report}")

def main() :
    try:
        selected_option = st.selectbox('Select an option', languages)
        selected_option_polly = st.selectbox('Select an option', merged_languages_with_voices[selected_option])
        file_uploaded = st.file_uploader('Upload the disaster image', type='jpg')

        print("File Uploaded:", file_uploaded)
        output, output2 = detect_objects_on_image(Image.open(file_uploaded))
        st.image(file_uploaded, use_column_width=True)

        output2 = [x for x in output2 if x[4] != 'Heavy Floods have occurred']

        image = draw_boxes_on_image(file_uploaded, output2)
        st.image(image, use_column_width=True)

        detected_objects = [i[4] for i in output2]

        st.write('Detected Objects : ', detected_objects)

        output_combined_report = generate_combined_report_from_csv('./keyword.csv', detected_objects)
        
        # AWS Translate
        result = translate.translate_text(Text=output_combined_report, SourceLanguageCode="en", TargetLanguageCode=language_codes[selected_option])
        print(result['TranslatedText'])

        # AWS Polly
        print("Selected Polly Voice:", selected_option_polly)
        print("Translated Text Length:", len(result['TranslatedText']))
        
        if len(result['TranslatedText']) >= 3000:
            result['TranslatedText'] = result['TranslatedText'][:2999]
        
        st.write('Report : ', result['TranslatedText'])
        print(result['TranslatedText'], len(result['TranslatedText']))
        
        response = polly.synthesize_speech(Engine='neural', Text=result['TranslatedText'], VoiceId=language_codes_polly[selected_option_polly], OutputFormat='mp3')
        body = response['AudioStream'].read()
        file_name = 'voice.mp3'

        with open("article.txt", "w", encoding="utf-8") as file:
            for text in result['TranslatedText']:
                report_sentences = result['TranslatedText'].split('\n')
                report_paragraphs = [' '.join(sentence.split()) for sentence in report_sentences if sentence.strip()]
                # Write the formatted text to the file
            file.write('\n\n'.join(report_paragraphs))
        
        with open(file_name, 'wb') as file:
            file.write(body)
        
        mp3_path = "voice.mp3"
        
        # if st.button('Play'):
        st.audio(mp3_path, format='audio/mp3', start_time=0)
        # Upload file to S3
        s3.upload_file("voice.mp3", "ccprojectnc", new_folder_name + "myvoice.mp3")
        s3.upload_file("article.txt", "ccprojectnc", new_folder_name + "article.txt")
    except  Exception as e:
        a = 10

footer = """<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}
</style>

<div class="footer">
<p align="center"> </p>
</div>
        """

st.markdown(footer, unsafe_allow_html = True)
if __name__ == '__main__' :
    main()