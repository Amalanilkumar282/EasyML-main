// // static/script.js
// document.addEventListener("DOMContentLoaded", function () {
//     const fileInput = document.querySelector('input[type="file"]');
//     const submitButton = document.querySelector('button[type="submit"]');
//     const predictButton = document.querySelector('button[type="predict"]');
//     const newValueInput = document.querySelector('input[name="new_value"]');
//     const predictionResult = document.getElementById('prediction-result');

//     // Disable predict button until a file is selected
//     predictButton.disabled = true;

//     fileInput.addEventListener("change", function () {
//         // Enable predict button when a file is selected
//         predictButton.disabled = false;
//     });

//     submitButton.addEventListener("click", function () {
//         // Display loading spinner while processing
//         this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
//         this.disabled = true;
//     });

//     predictButton.addEventListener("click", function () {
//         // Display loading spinner while predicting
//         this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
//         this.disabled = true;

//         // Clear previous prediction result
//         predictionResult.innerHTML = '';

//         // Perform AJAX request to get the prediction result
//         const xhr = new XMLHttpRequest();
//         xhr.open("POST", "/predict", true);
//         xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");

//         // Send the new value and model parameters to the server for prediction
//         xhr.send(`new_value=${newValueInput.value}&model=${predictButton.dataset.model}`);
//     });
// });
