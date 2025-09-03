import os
import smtplib
import ssl
from email.message import EmailMessage

from activetigger.config import config
from activetigger.db.manager import DatabaseManager
from activetigger.db.models import Messages


class Messages:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    mail_available: bool = config.mail_available
    mail_server: str | None = config.mail_server
    mail_server_port: int = config.mail_server_port
    mail_account: str | None = config.mail_account
    mail_password: str | None = config.mail_password

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        if self.mail_available:
            print("Mail service is available")
        else:
            print("Mail service is not available")

    def send_mail(self, to: str, subject: str, body: str) -> bool:
        """
        Send a mail
        """
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"Active Tigger <{self.mail_account}>"
        msg["To"] = to
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.mail_server, self.mail_server_port, context=context) as server:
            server.login(self.mail_account, self.mail_password)
            result = server.send_message(msg)

        if result != {}:
            print(f"Failed to send email to {to}: {result}")

    def send_mail_reset_password(self, user_name: str, mail: str, new_password: str) -> None:
        """
        Send a mail to reset the password
        """
        if not self.mail_available:
            raise Exception("Mail service is not available")

        subject = "Active Tigger - Password Reset"
        body = f"""
        Hello,

        Your password has been reset for the account : {user_name}
        
        Your new password is: {new_password}

        Please log in and change your password as soon as possible.

        Best regards,
        The Active Tigger Team
        """
        self.send_mail(mail, subject, body)

    def get_messages_system(self) -> list[Messages]:
        """
        Get all system messages ordered by time desc
        """
        return self.db_manager.messages_service.get_messages_system()

    def get_messages_for_project(self, project_slug: str) -> list[Messages]:
        """
        Get all project messages for a specific project ordered by time desc
        """
        return self.db_manager.messages_service.get_messages_for_project(project_slug)

    def get_messages_for_user(self, user_name: str) -> list[Messages]:
        """
        Get all user messages for a specific user ordered by time desc
        """
        return self.db_manager.messages_service.get_messages_for_user(user_name)
