import hmac
import requests
import json
import streamlit as st
import os
from test import get_features
from config import keys


# Set your OpenAI API key here
OPENAI_API_KEY,ASSISTANT_ID=keys()

# Headers for OpenAI API requests
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v1"
}



def add_to_thread(api_key,thread_id,message):
    url = "https://api.openai.com/v1/threads/"+thread_id+"/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "assistants=v1"
    }
    data = {
        "role": "user",
        "content": message
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


def execute_command(api_key, assistant_id,thread_id):
    url = "https://api.openai.com/v1/threads/"+thread_id +"/runs"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v1"
    }
    data = {
        "assistant_id": assistant_id
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response= response.json()
    print("Execute Command response" ,response  )
    return response["id"]


def get_run_update(api_key, thread_id, run_id):
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "assistants=v1"
    }
    response1 = requests.get(url, headers=headers)
    response= response1.json()
    #print("get_run_update response" , response["status"])
    if response["status"] == "failed":
        print(response1.json())
    return response["status"]

def retrieve_function(api_key, thread_id, run_id):
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "assistants=v1"
    }
    response = requests.get(url, headers=headers)
    response = response.json()
    return response


def get_openai_thread_messages(api_key, thread_id):
    url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "assistants=v1"
    }
    response = requests.get(url, headers=headers)
    response = response.json()
    print(response)
    convolist = []
    for message in response["data"]:
        role = message["role"]
        created_at = message["created_at"]
        value = message["content"][0]["text"]["value"]
        convolist.append({"role": role, "created_at": created_at, "message": value})

    return convolist

# Function to get the chat history from the API
def get_chat_history(thread_id):
    response = requests.get(f"https://api.openai.com/v1/threads/{thread_id}/messages", headers=headers)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.error("Failed to fetch chat history")
        return []

def submit_tool_outputs(api_key, thread_id, run_id, tool_outputs):
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs"
    print("submiting to openai")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v1"
    }
    data = {
        "tool_outputs": tool_outputs
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print("done")
    return response.json()

def send_to_ai(api_key,thread_id,message,assistant_id):
    add_to_thread(api_key,
                  thread_id,message)

    execution_id=execute_command(api_key,assistant_id,thread_id)
    status=get_run_update(api_key,thread_id,execution_id)
    while status == "in_progress" or status == "queued":
        print("in queued state")
        status = get_run_update(api_key,
                                   thread_id,execution_id)

        print("status " , status)

        if status == "requires_action":
           print("Requires action")
           data= retrieve_function(api_key,thread_id,execution_id)
           #print("data = ", data)
           # Extract tool calls
           tool_calls_info = {}
           for tool_call in data["required_action"]["submit_tool_outputs"]["tool_calls"]:
               call_id = tool_call["id"]
               function_name = tool_call["function"]["name"]
               arguments = tool_call["function"]["arguments"]
               tool_calls_info[call_id] = {"function_name": function_name, "arguments": arguments,"call_id" : call_id}

           # The dictionary now contains the extracted information
           print("tools info",tool_calls_info)

           if function_name=="get_features":
               features=get_features()
               print("retreieving the data")
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(features)}
               ]
               print(features)           
           elif function_name=="list_doctors":
               print("retreieving all available doctors")
               args = json.loads(arguments)  # Assuming arguments is a JSON string
               specialist = args.get("specialist")
               clinic_id=args.get("clinic_id")
               doctors=get_doctors(specialist,clinic_id)
               print(doctors)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(doctors)}
               ]
           elif function_name=="list_slot":
               print("retreieving all available slots")
               args = json.loads(arguments)  # Assuming arguments is a JSON string
               doc_id = args.get("practitioner_id")
               clinic_id=args.get("clinic_id")
               date=args.get("date")
               slots=get_slots(doc_id,clinic_id,date)
               print(slots)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(slots)}
               ]
           elif function_name=="book_appoinment":
               print("Booking appoinment")
               args = json.loads(arguments)
               patient_id="SDP10646"
              #patient_id = args.get("patient_id")
               clinic_id=args.get("clinic_id")               
               doc_id = args.get("practitioner_id")
               token=args.get("token")
               date=args.get("date")
               time=args.get("time")
               type_id=args.get("type_id")
               infos=book_appoinment(patient_id,doc_id,clinic_id,token,date,time,type_id)
               print(infos)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(infos)}
               ]


           elif function_name=="get_all_available_specialists":
               print("retreieving all available specialists")
               specialists=get_specialists()
               print(specialists)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(specialists)}
               ]

           elif function_name == "get_prescription":
               print("retrieving prescription")
               # Assuming the appointment_id is passed as an argument in the function call
               appointment_id = json.loads(arguments).get("appointment_id")

               prescription_info = get_prescription_url(appointment_id)  # Call the function from test.py
               print(prescription_info)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(prescription_info)
                   }
               ]
           elif function_name == "list_health_package_appointments":
               print("listing checkup appointments")
               # Extracting parameters from the arguments
               args = json.loads(arguments)
               #user_id = args.get("user_id")
               user_id="SDP4809"
               start_date = args.get("start_date", None)  # Optional, default to None if not provided
               end_date = args.get("end_date", None)  # Optional, default to None if not provided

               # Assuming get_checkup_appointments is a function you've defined or will define
               appointments_info = get_lab_appointments_package_names(user_id)
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": str(appointments_info)
                   }
               ]
           elif function_name == "list_health_packages":
               print("Retrieving health packages based on specified criteria")
               args = json.loads(arguments)  # Assuming arguments is a JSON string
               city = args.get("city", "Kochi")  # Default to Kochi if city not provided
               category = args.get("category", None)
               lab_name = args.get("lab_name", None)

               packages_info = get_packages()
               tool_outputs = [
                   {
                       "tool_call_id": call_id,
                       "output": json.dumps(packages_info)
                   }
               ]

           submit_tool_outputs(api_key,thread_id,execution_id,tool_outputs)
           print("successfully submitted to ai")

           status = get_run_update(api_key,
                                   thread_id, execution_id)












    #convo_list=get_openai_thread_messages(api_key="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",thread_id="thread_pZ1PSPSUitgF1oAPlHa5rDu2")


    #return convo_list

# Function to create a new thread
def create_thread():
    response = requests.post("https://api.openai.com/v1/threads", headers=headers)
    return response.json()["id"]

# Function to load thread_id from a file
def load_thread_id():
    if os.path.exists('thread_id_noco.json'):
        with open('thread_id_noco.json', 'r') as file:
            data = json.load(file)
            return data.get('thread_id')
    return None

# Function to save thread_id to a file
def save_thread_id(thread_id):
    with open('thread_id_noco.json', 'w') as file:
        json.dump({'thread_id': thread_id}, file)

# Function to clear thread_id from the file
def delete_thread_id():
    if os.path.exists('thread_id_noco.json'):
        os.remove('thread_id_noco.json')


def clear_chat(thread_id):
    response = requests.delete(f"https://api.openai.com/v1/threads/{thread_id}", headers=headers)
    if response.status_code == 200:
        st.sidebar.success("Chat cleared successfully.")
        return True
    else:
        st.sidebar.error("Failed to clear chat.")
        return False


# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # Don't store the password.
#         else:
#             st.session_state["password_correct"] = False

#     # Return True if the password is validated.
#     if st.session_state.get("password_correct", False):
#         return True

#     # Show input for password.
#     st.text_input(
#         "Password", type="password", on_change=password_entered, key="password"
#     )
#     if "password_correct" in st.session_state:
#         st.error("😕 Password incorrect")
#     return False

def fetch_file_names(assistant_id):
    # First, fetch the file IDs associated with the assistant
    assistant_files_response = requests.get(
        f"https://api.openai.com/v1/assistants/{assistant_id}/files",
        headers=headers
    )
    if assistant_files_response.status_code != 200:
        st.sidebar.error("Failed to fetch assistant files.")
        return []

    assistant_file_ids = [f["id"] for f in assistant_files_response.json()["data"]]

    # Now, fetch all files to get filenames
    all_files_response = requests.get("https://api.openai.com/v1/files", headers=headers)
    if all_files_response.status_code != 200:
        st.sidebar.error("Failed to fetch file details.")
        return []

    all_files = all_files_response.json()["data"]

    # Map file IDs from the assistant to filenames
    file_details = [(file["id"], file["filename"]) for file in all_files if file["id"] in assistant_file_ids]

    return file_details

# Main function to handle the Streamlit app
def main():
    st.title("NoCo The Bot")




    # This key should ideally be secured
    api_key = OPENAI_API_KEY
    #thread_id = "thread_tyryBrOLVFSBJViXIOLohH55"
    assistant_id = ASSISTANT_ID
    #convos=get_openai_thread_messages(api_key,thread_id)
    #display_convos(convos)

    # if check_password():
    with st.sidebar:

        if st.sidebar.button("Clear Chat"):
            if clear_chat(st.session_state.thread_id):
                # Reset the chat history in the UI
                st.session_state.chat_history = []
                delete_thread_id()
                # Remove thread_id from session state
                st.session_state.thread_id = None

        st.title("Copilot Files")
        file_details = fetch_file_names(ASSISTANT_ID)
        for file_id, filename in file_details:
            col1, col2 = st.sidebar.columns([4, 1])
            col1.text(filename)



    # Load or create thread_id
    thread_id = load_thread_id()
    if thread_id is None:
        thread_id = create_thread()
        save_thread_id(thread_id)
        st.session_state.thread_id = thread_id
    else:
        st.session_state.thread_id = thread_id

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input()
    if user_input:
        send_to_ai(api_key, thread_id, user_input, assistant_id)
    if st.session_state.thread_id != None:
        chat_history = get_chat_history(st.session_state.thread_id)
        #print(st.session_state.thread_id)

        for message in reversed(chat_history):
            role = message["role"]
            text = message["content"][0]["text"]["value"]
            st.chat_message(role).write(text)



# Run the app
if __name__ == "__main__":
    main()