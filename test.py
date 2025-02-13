import requests
import json
import uuid
import os
from dotenv import load_dotenv
load_dotenv() 

# Zendesk Configuration
subdomain = os.environ.get("ZENDESK_SUBDOMAIN")  # Replace with your Zendesk subdomain
admin_email =os.environ.get("ZENDESK_EMAIL")  # Replace with your Zendesk agent email
api_token = os.environ.get("ZENDESK_API_TOKEN")  # Replace with your Zendesk API token

# API Setup
base_url = f"https://{subdomain}.zendesk.com"
auth = (f"{admin_email}/token", api_token)
headers = {"Content-Type": "application/json"}

TICKET_ID = 1

# Create Temporary Anonymous User and Ticket
def create_anonymous_ticket(message):
    # Generate unique temporary identifier
    temp_email = f"anonymous_{uuid.uuid4().hex}@gmail.com"
    temp_name = "Test User from Ai Department Web"

    ticket_data = {
        "request": {
            "subject": "Test Support Request from AI Department",
            "comment": {"body": message},
            "requester": {
                "name": temp_name,
                "email": temp_email
            }
        }
    }

    response = requests.post(
        f"{base_url}/api/v2/requests",
        auth=auth,
        headers=headers,
        data=json.dumps(ticket_data)
    )

    if response.status_code == 201:
        ticket = response.json()["request"]
        print(f"Ticket created: {ticket['id']}")
        TICKET_ID = ticket['id']
        return ticket["requester_id"], ticket["id"]
    else:
        print(f"Error creating ticket: {response.text}")
        return None, None

def update_ticket_status( new_status="open"):
    ticket_data = {
        "ticket": {
            "status": new_status
        }
    }

    response = requests.put(
        f"{base_url}/api/v2/tickets/{TICKET_ID}",
        auth=auth,
        headers=headers,
        data=json.dumps(ticket_data)
    )

    if response.status_code == 200:
        print(f"Ticket {TICKET_ID} status updated to {new_status}.")
        return True
    else:
        print(f"Error updating ticket status: {response.text}")
        return False


# Update User Details
def update_user_details(requester_id, new_email, new_name):
    # Check if the user with the email already exists
    search_response = requests.get(
        f"{base_url}/api/v2/users?email={new_email}",
        auth=auth,
        headers=headers
    )

    if search_response.status_code == 200:
        users = search_response.json().get("users", [])
        if users:  # User exists
            print(f"User with email {new_email} already exists.")
            # update the ticket status to open
            update_ticket_status("open")
        else:
            print(f"No user found with email {new_email}, proceeding to create/update.")

    # Prepare user data for updating
    user_data = {
        "user": {
            "email": new_email,
            "name": new_name
        }
    }

    # Send the PUT request to update the user
    response = requests.put(
        f"{base_url}/api/v2/users/{requester_id}",
        auth=auth,
        headers=headers,
        data=json.dumps(user_data)
    )

    if response.status_code == 200:
        print("User updated successfully")
        return True
    else:
        print(f"Error updating user: {response.text}")
        return False

# Main Workflow
def main():
    # Step 1: Create anonymous ticket
    requester_id, ticket_id = create_anonymous_ticket("Hello, this is just a test. Ignore Please.")
    if not requester_id:
        return

    # Step 2: Collect user details (simulated input)
    print("\nPlease provide your contact information:")
    new_email = input("Email address: ").strip()
    new_name = input("Full name: ").strip()

    # Step 3: Update user details
    if update_user_details(requester_id, new_email, new_name):
        print(f"Ticket {ticket_id} now associated with {new_email}")
    else:
        print("Failed to update user details")

if __name__ == "__main__":
    main()