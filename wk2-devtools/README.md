# Clinic Appointment Management App

## The Challenge: Waiting Times
Even with an appointment, waiting at a clinic or hospital is often an unavoidable and unwelcome experience. It's the last place anyone wants to spend their time, especially when feeling unwell. The uncertainty of wait times can lead to unnecessary stress and wasted hours.

## Our Solution: Peer-to-Peer Appointment Management
This Flask web application addresses the problem of prolonged waiting times by enabling peer-to-peer management of appointments and wait times. The goal is to empower patients and staff to collaboratively track progress and update appointment statuses, ultimately allowing patients to leave home just in time to meet their doctor, rather than waiting in the clinic.

## What the App Does:
*   **Patient Register:** Maintains a list of today's patient appointments, including patient name, scheduled time, current wait time, status (waiting/completed), location, distance, and recommended start-from-home time.
*   **Dynamic Wait Time Updates:** When an appointment is marked as completed, the wait times for all subsequent waiting patients are automatically reduced.
*   **Appointment Completion:** Each patient entry has a button to mark their appointment as completed.
*   **New Patient Addition:** A simple form allows staff to add new patients to the daily register, with their name and scheduled time.
*   **Real-time UI:** The user interface updates dynamically using AJAX/Flask routes to reflect changes in appointment statuses and wait times.
*   **In-Memory Data:** All appointment data is maintained in server memory for this prototype (no database required).
*   **Clear Instructions:** Provides basic instructions on the main page for both patients and staff.

## How to Run the App Locally:

1.  **Navigate to the project directory:**

    ```bash
    cd /Users/vinodjoshi/erav4/wk2-devtools/
    ```

2.  **Run the Flask application:**

    ```bash
    python main.py
    ```

3.  **Open your web browser** and go to:

    ```
    http://127.0.0.1:5000/
    ```

4.  To **close the app**, go back to your terminal and press `Ctrl + C`.
