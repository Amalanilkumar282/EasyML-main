<?php
$servername = "localhost";
$username = "root";
$password = "password";
$dbname = "NoCodeML";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get username and password from the form
$email = $_POST["email"];
$password = $_POST["password"];

// Validate login
$sql = "SELECT * FROM USERS WHERE email='$email' AND password='$password'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // Login successful
    $response = array("success" => true);
} else {
    // Login failed
    $response = array("success" => false);
}

echo json_encode($response);

$conn->close();
?>
