import React, { useState } from "react";

const Home = () => {
  const [image, setImage] = useState(null); // To store the selected image
  const [preview, setPreview] = useState(null); // To display the image preview
  const [result, setResult] = useState(null); // To display the result from the FastAPI backend

  // Function to handle image selection
  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file)); // Create a preview URL for the selected image
    }
  };

  // Function to handle form submission
  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data); // Set the result data from the backend
      } else {
        console.error("Error uploading the image.");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white font-mono">
      <div
        className="bg-gray-800 p-8 rounded-lg shadow-lg transform transition-transform hover:scale-105"
        style={{ width: "400px" }}
      >
        <h1 className="text-3xl font-bold mb-4">Potato Disease Prediction</h1>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        {preview && (
          <img src={preview} alt="Uploaded Preview" width="400" height="300" />
        )}
        <button
          onClick={handleSubmit}
          className="mt-3 w-full py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-semibold transition-all"
        >
          Submit
        </button>
        {result && (
          <div className="mt-4 p-4 bg-gray-700 rounded-lg text-sm text-gray-300">
            <h2 className="text-2xl">Prediction Result</h2>
            <p className="text-xl">Disease: {result.class}</p>
            <p className="text-xl">
              Confidence: {result.confidence.toFixed(2) * 100}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;
