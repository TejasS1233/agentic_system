import json
from faker import Faker

fake = Faker()

def generate_test_users(num_users):
    users = []
    for _ in range(num_users):
        user = {
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address()
        }
        users.append(user)
    return users

def save_as_json(users, filename):
    with open(filename, 'w') as f:
        json.dump(users, f, indent=4)

def validate_json(filename):
    try:
        with open(filename, 'r') as f:
            json.load(f)
        print("JSON is valid.")
    except json.JSONDecodeError as e:
        print(f"JSON is not valid: {e}")

if __name__ == "__main__":
    num_users = 50
    filename = "test_users.json"
    users = generate_test_users(num_users)
    save_as_json(users, filename)
    validate_json(filename)