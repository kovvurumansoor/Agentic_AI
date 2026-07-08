from utils.llm import llm


def generate_email():
    prompt = """
    Write a professional email to a professor requesting
    a 2-day extension for project submission because of health issues.
    """

    response = llm.invoke(prompt)

    return response.content


def human_approval(email):
    print("\n========== EMAIL DRAFT ==========\n")
    print(email)
    print("\n================================")

    decision = input("\nType APPROVED to send the email: ")

    return decision.upper() == "APPROVED"


def send_email(email):
    print("\n📤 Sending Email...\n")

    print(email)

    print("\n✅ Email Sent Successfully!")


def main():
    email = generate_email()

    approved = human_approval(email)

    if approved:
        send_email(email)
    else:
        print("\n❌ Email Cancelled by User.")


if __name__ == "__main__":
    main()