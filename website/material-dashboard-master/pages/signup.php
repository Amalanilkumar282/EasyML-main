<?php
// Assuming you have a MySQL database connection
$servername = "your_database_server";
$username = "your_database_username";
$password = "your_database_password";
$dbname = "YOUR_DATABASE_NAME";

// Create a connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Initialize response array
$response = array('success' => false, 'message' => '');

// Process form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $full_name = $_POST["full_name"];
    $phone_number = $_POST["phone_number"];
    $dob = $_POST["dob"];
    $email = $_POST["email"];
    $password = password_hash($_POST["password"], PASSWORD_DEFAULT); // Hash the password before storing

    // Insert data into the USERS table
    $sql = "INSERT INTO USERS (full_name, phone_number, dob, email, password) VALUES ('$full_name', '$phone_number', '$dob', '$email', '$password')";

    if ($conn->query($sql) === TRUE) {
        $response['success'] = true;
        $response['message'] = 'Account created successfully';
    } else {
        $response['message'] = 'Error: ' . $sql . '<br>' . $conn->error;
    }
}

// Close the connection
$conn->close();

// Send JSON response to the client
header('Content-Type: application/json');
echo json_encode($response);
?>
