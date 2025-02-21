from pydantic import BaseModel
import requests
import json
import uuid
import os
from dotenv import load_dotenv
from typing import Dict, Tuple, Optional

# Load environment variables
load_dotenv()

class BrowserInfo(BaseModel):
    browser_family: str
    browser_version: Optional[str]
    os_family: str
    os_version: Optional[str]
    device_family: str
    device_brand: Optional[str]
    device_model: Optional[str]
    is_mobile: bool
    is_tablet: bool
    is_desktop: bool
    is_bot: bool
    raw_user_agent: str

class LocationInfo(BaseModel):
    country_code: Optional[str]
    country_name: Optional[str]
    city: Optional[str]
    postal_code: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    continent: Optional[str]
    subdivision: Optional[str]
    accuracy_radius: Optional[int]

class RequestInfo(BaseModel):
    # Previous request fields...
    method: str
    url: str
    base_url: str
    path: str
    headers: Dict[str, str]
    client_host: Optional[str]
    
    # New fields for browser and location
    browser_info: Optional[BrowserInfo]
    location_info: Optional[LocationInfo]

class ZendeskTicketManager:
    def __init__(self):
        """Initialize Zendesk configuration from environment variables."""
        self.subdomain = os.environ.get("ZENDESK_SUBDOMAIN")
        self.admin_email = os.environ.get("ZENDESK_EMAIL")
        self.api_token = os.environ.get("ZENDESK_API_TOKEN")
        
        if not all([self.subdomain, self.admin_email, self.api_token]):
            raise ValueError("Missing required environment variables")
        
        self.base_url = f"https://{self.subdomain}.zendesk.com"
        self.auth = (f"{self.admin_email}/token", self.api_token)
        self.headers = {"Content-Type": "application/json"}
        self.current_ticket_id = None

    def create_anonymous_ticket(self, message: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Create a temporary anonymous ticket in Zendesk with request metadata.
        
        Args:
            message (str): Initial ticket message
            request_info (RequestInfo): Request metadata including browser and location info
            
        Returns:
            Tuple[Optional[int], Optional[int]]: Tuple of (requester_id, ticket_id)
        """
        temp_email = f"anonymous_{uuid.uuid4().hex}@temporary.com"
        temp_name = "Anonymous User"
        
        # Create metadata dictionary
        # metadata = {
        #     "custom": {
        #         "request": {
        #             "method": request_info.method,
        #             "url": request_info.url,
        #             "base_url": request_info.base_url,
        #             "path": request_info.path,
        #             "client_host": request_info.client_host
        #         }
        #     }
        # }
        
        # Add browser info if available
        # if request_info.browser_info:
        #     metadata["custom"]["browser"] = {
        #         "family": request_info.browser_info.browser_family,
        #         "version": request_info.browser_info.browser_version,
        #         "os_family": request_info.browser_info.os_family,
        #         "os_version": request_info.browser_info.os_version,
        #         "device_family": request_info.browser_info.device_family,
        #         "device_brand": request_info.browser_info.device_brand,
        #         "device_model": request_info.browser_info.device_model,
        #         "is_mobile": request_info.browser_info.is_mobile,
        #         "is_tablet": request_info.browser_info.is_tablet,
        #         "is_desktop": request_info.browser_info.is_desktop,
        #         "is_bot": request_info.browser_info.is_bot,
        #         "user_agent": request_info.browser_info.raw_user_agent
        #     }
        
        # Add location info if available
        # if request_info.location_info:
        #     metadata["custom"]["location"] = {
        #         "country_code": request_info.location_info.country_code,
        #         "country_name": request_info.location_info.country_name,
        #         "city": request_info.location_info.city,
        #         "postal_code": request_info.location_info.postal_code,
        #         "latitude": request_info.location_info.latitude,
        #         "longitude": request_info.location_info.longitude,
        #         "timezone": request_info.location_info.timezone,
        #         "continent": request_info.location_info.continent,
        #         "subdivision": request_info.location_info.subdivision,
        #         "accuracy_radius": request_info.location_info.accuracy_radius
        #     }

        ticket_data = {
            "request": {
                "subject": "Support Request from Shopify ES",
                "comment": {
                    "body": "SanaExpert - Support Team.",
                    "author_id": "31549253490321",
                    "public": False,
                },
                "requester": {
                    "name": temp_name
                },
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v2/requests",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            
            ticket = response.json()["request"]

            #update the ticket status to pending
            # ticket_data = {
            #         "ticket": {
            #             "status": "pending",
            #             "tags": ["ticket_by_ai"]
            #         }
            #     }
            # response = requests.put(
            #     f"{self.base_url}/api/v2/tickets/{ticket['id']}",
            #     auth=self.auth,
            #     headers=self.headers,
            #     json=ticket_data
            # )
            # response.raise_for_status()
            self.current_ticket_id = ticket['id']
            return ticket["requester_id"], ticket["id"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating ticket: {str(e)}")
            return None, None

    def update_ticket_status(self, ticket_id: int, new_status: str = "open") -> bool:
        """
        Update the status of a specified ticket.
        
        Args:
            ticket_id (int): ID of the ticket to update
            new_status (str): New status to set (default: "open")
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "status": new_status
            }
        }
        #assign to omer jadoon, clara
        ticket_data["ticket"]["assignee_id"] = "25793382446353"
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket status: {str(e)}")
            return False
        
    def assign_ticket(self, ticket_id: int, requester_id: int) -> bool:
        """
        Assign a ticket to a specific requester.
        
        Args:
            ticket_id (int): ID of the ticket to update
            requester_id (int): ID of the requester to assign the ticket to
            
        Returns:
            bool: Success status of the update
        """
        ticket_data = {
            "ticket": {
                "requester_id": requester_id
            }
        }

        #TODO modify the ticket to assign the comments to the requester as well.
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=ticket_data
            )
            response.raise_for_status()
            print(f"Successfully assigned ticket {ticket_id} to requester {requester_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error assigning ticket: {str(e)}")
            return False

    def update_user_details(self, requester_id: int,ticket_id:int, new_email: str, new_name: str) -> bool:
        """
        Update user details and handle existing user cases.
        
        Args:
            requester_id (int): ID of the requester to update
            new_email (str): New email address
            new_name (str): New full name
            
        Returns:
            bool: Success status of the update
        """
        try:
            # Check if user already exists
            search_response = requests.get(
                f"{self.base_url}/api/v2/users/search?query={new_email}",
                auth=self.auth,
                headers=self.headers
            )
            search_response.raise_for_status()
            
            existing_users = search_response.json().get("users", [])
            
            if existing_users:
                existing_user = existing_users[0]
                print(f"User found: {existing_user['name']} ({existing_user['email']})")
                
                # If we have a current ticket, update its status 

                ### update the requester_id to existing ticket
                
                self.assign_ticket(ticket_id, existing_user['id'])
                self.update_ticket_status(ticket_id, "open")
                return True
            
            # Update user details if no existing user found
            user_data = {
                "user": {
                    "email": new_email,
                    "name": new_name
                }
            }
            
            update_response = requests.put(
                f"{self.base_url}/api/v2/users/{requester_id}",
                auth=self.auth,
                headers=self.headers,
                json=user_data
            )
            update_response.raise_for_status()
            self.update_ticket_status(ticket_id, "open")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating user: {str(e)}")
            return False

    def add_public_comment(self, ticket_id: int, comment: str, requester_id: str) -> bool:
        """
        Add a public comment to a Zendesk ticket.
        
        Args:
            ticket_id (int): The ID of the ticket to update.
            comment (str): The comment text.
            as_bot (bool): Whether the comment should be posted as a bot (True) or as the customer (False).
        
        Returns:
            bool: Success status of the comment addition.
        """
        comment_data = {
            "ticket": {
                "comment": {
                    "body": comment,
                    "public": False,
                }
            },
            "tags": ["ticket_by_ai"]
        }
        
        if requester_id != "32601040249617":
            comment_data["ticket"]["comment"]["author_id"] = requester_id
        else:
            comment_data["ticket"]["comment"]["author_id"] = "32601040249617"  # Zendesk automatically assigns the agent/bot
        
        try:
            response = requests.put(
                f"{self.base_url}/api/v2/tickets/{ticket_id}",
                auth=self.auth,
                headers=self.headers,
                json=comment_data
            )
            response.raise_for_status()
            print(f"Successfully added public comment to ticket {ticket_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Error adding public comment: {str(e)}")
            return False




def main():
    """Main workflow for ticket creation and user management."""
    try:
        # Initialize the ticket manager
        manager = ZendeskTicketManager()
        
        # Step 1: Create anonymous ticket
        requester_id, ticket_id = manager.create_anonymous_ticket(
            "Initial support request"
        )
        
        if not requester_id:
            print("Failed to create ticket. Exiting.")
            return
        
        # Step 2: Get user details
        print("\nPlease provide your contact information:")
        new_email = input("Email address: ").strip()
        new_name = input("Full name: ").strip()
        
        # Validate input
        if not all([new_email, new_name]):
            print("Email and name are required. Exiting.")
            return
        
        # Step 3: Update user details
        if manager.update_user_details(requester_id,ticket_id, new_email, new_name):
            print(f"Successfully processed ticket {ticket_id} for {new_email}")
        else:
            print("Failed to update user details")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()