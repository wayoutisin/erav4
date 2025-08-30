
from flask import Flask, render_template, request, jsonify
import datetime

app = Flask(__name__)

# In-memory data store for appointments
appointments = [
    {
        "id": 1,
        "name": "Alice Smith",
        "scheduled_time": "09:00",
        "wait_time": 10,
        "status": "waiting",
        "location": "123 Main St, Anytown",
        "distance": "5 miles",
        "start_from_home_time": "08:30",
    },
    {
        "id": 2,
        "name": "Bob Johnson",
        "scheduled_time": "09:30",
        "wait_time": 25,
        "status": "waiting",
        "location": "456 Oak Ave, Somewhere",
        "distance": "10 miles",
        "start_from_home_time": "08:50",
    },
    {
        "id": 3,
        "name": "Charlie Brown",
        "scheduled_time": "10:00",
        "wait_time": 40,
        "status": "waiting",
        "location": "789 Pine Ln, Nowhere",
        "distance": "2 miles",
        "start_from_home_time": "09:40",
    },
]

def calculate_wait_time(scheduled_time_str):
    # This is a simplified wait time calculation for demonstration.
    # In a real app, this would be more complex.
    current_time = datetime.datetime.now().strftime("%H:%M")
    return f"Approx. {len([appt for appt in appointments if appt['status'] == 'waiting']) * 15} minutes"


@app.route("/")
def index():
    return render_template("index.html", appointments=appointments)


@app.route("/complete_appointment/<int:appointment_id>", methods=["POST"])
def complete_appointment(appointment_id):
    for appt in appointments:
        if appt["id"] == appointment_id:
            appt["status"] = "completed"
            # Decrease wait times for remaining patients
            for remaining_appt in appointments:
                if remaining_appt["status"] == "waiting":
                    # Simplistic wait time reduction for demo purposes
                    # In a real app, this would involve more sophisticated logic
                    remaining_appt["wait_time"] = max(0, remaining_appt["wait_time"] - 5)
            return jsonify({"success": True})
    return jsonify({"success": False, "message": "Appointment not found"}), 404


@app.route("/add_appointment", methods=["POST"])
def add_appointment():
    data = request.get_json()
    patient_name = data.get("name")
    scheduled_time = data.get("scheduled_time")

    if not patient_name or not scheduled_time:
        return jsonify({"success": False, "message": "Name and scheduled time are required"}), 400

    new_id = max([appt["id"] for appt in appointments]) + 1 if appointments else 1
    new_appointment = {
        "id": new_id,
        "name": patient_name,
        "scheduled_time": scheduled_time,
        "wait_time": len([appt for appt in appointments if appt['status'] == 'waiting']) * 15,
        "status": "waiting",
        "location": "Random Clinic Location",  # Hardcoded for new appointments
        "distance": "Approx. 7 miles",  # Hardcoded for new appointments
        "start_from_home_time": "Varies",  # Placeholder, could be calculated
    }
    appointments.append(new_appointment)

    return jsonify({"success": True, "appointment": new_appointment})


@app.route("/get_appointments")
def get_appointments():
    return jsonify(appointments)


if __name__ == "__main__":
    app.run(debug=True)
