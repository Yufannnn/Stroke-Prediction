document.addEventListener("DOMContentLoaded", function() { // Make sure DOM is fully loaded
    document.getElementById("submit-button").addEventListener("click", function(event) {
        event.preventDefault(); // prevent the form from submitting and the page from reloading
        document.querySelector(".result-section").style.display = "block";
    });
});
