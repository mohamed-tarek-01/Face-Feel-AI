document.addEventListener("DOMContentLoaded", function () {
    const imageUpload = document.getElementById("imageUpload");
    const predictBtn = document.getElementById("predictBtn");
    const processedImage = document.getElementById("processedImage");
    const startCamera = document.getElementById("startCamera");
    const stopCamera = document.getElementById("stopCamera");
    const videoStream = document.getElementById("videoStream");
    const canvas = document.createElement("canvas");
    let streaming = false;
    let stream = null;

    // Handle image upload and display it
    imageUpload.addEventListener("change", function (event) {
        let reader = new FileReader();
        reader.onload = function () {
            processedImage.src = reader.result;
            processedImage.classList.remove("d-none"); // Show the image
        };
        reader.readAsDataURL(event.target.files[0]);
    });

    // Send image to the server and display the processed result
    predictBtn.addEventListener("click", function () {
        let file = imageUpload.files[0];
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        let formData = new FormData();
        formData.append("file", file);

        fetch("/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.blob())
            .then((blob) => {
                let imageUrl = URL.createObjectURL(blob);
                processedImage.src = imageUrl;
                processedImage.classList.remove("d-none"); // Show the processed image
            })
            .catch((error) => console.error("Error:", error));
    });

    // Start the camera and capture frames
    startCamera.addEventListener("click", function () {
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then(function (videoStreamObj) {
                stream = videoStreamObj;
                videoStream.srcObject = stream;
                videoStream.classList.remove("d-none"); // Show video stream
                startCamera.classList.add("d-none");
                stopCamera.classList.remove("d-none");
                streaming = true;
                captureFrames(); // Start capturing frames
            })
            .catch(function (error) {
                console.error("Error accessing the camera: ", error);
            });
    });

    // Stop the camera and hide video stream
    stopCamera.addEventListener("click", function () {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop()); // Stop camera stream
        }
        videoStream.classList.add("d-none");
        startCamera.classList.remove("d-none");
        stopCamera.classList.add("d-none");
        streaming = false;
    });

    // Capture frames from the camera and send them to the server
    function captureFrames() {
        if (!streaming) return;

        let context = canvas.getContext("2d");
        canvas.width = videoStream.videoWidth;
        canvas.height = videoStream.videoHeight;
        context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            let formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            fetch("/predict", {
                method: "POST",
                body: formData,
            })
                .then((response) => response.blob())
                .then((blob) => {
                    let imageUrl = URL.createObjectURL(blob);
                    processedImage.src = imageUrl;
                    processedImage.classList.remove("d-none"); // Show processed frame
                })
                .catch((error) => console.error("Error:", error));
        }, "image/jpeg");

        setTimeout(captureFrames, 1000); // Repeat every second
    }
});
