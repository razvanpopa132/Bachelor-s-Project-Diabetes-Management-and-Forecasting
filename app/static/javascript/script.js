function showContent(option) {
  var content = document.getElementById("content");

  if (option === "Incarca CSV") {
    content.innerHTML = `
      <h1>${option}</h1>
      <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" id="csvFile" accept=".csv">
    <button type="button" onclick="uploadCSV(); showUploadStatus()">Upload</button>
    <div id="status"></div>
    
</form>

<div id="messageContainer"></div>
    `;

    var uploadForm = document.getElementById("uploadForm");
    uploadForm.addEventListener("submit", uploadCSV);
  } else if (option === "Creeaza grafic") {
    content.innerHTML = `
    <div class="dates">
  <form id="graphForm" method="POST">
    <label for="start">Dată de început:</label>
    <input type="date" id="start" name="start_date" value="2019-07-22" />

    <label for="end">Dată de sfârșit:</label>
    <input type="date" id="end" name="end_date" value="2019-07-23" />

    <label for="minimum-glucose">Prag glicemie minimă</label>
    <input type="number" id="minimum-glucose" name="min_glucose" />

    <label for="maximum-glucose">Prag glicemie maximă</label>
    <input type="number" id="maximum-glucose" name="max_glucose" />

    <button type="button" class="submit-btn" onclick="submitForm(); showUploadStatus()">Creează grafic</button>
    <div id="status"></div>
    </form>
  <div id="graphContainer"></div>
  
</div>
`;
  } else if (option === "Realizeaza predictii") {
    content.innerHTML = `
    <div class="dates">
  <form id="graphForm1" method="POST">
    <label for="start">Dată de început:</label>
    <input type="date" id="start" name="start_date" value="2019-07-22" />

    <label for="end">Dată de sfârșit:</label>
    <input type="date" id="end" name="end_date" value="2019-07-23" />


    <button type="button" class="submit-btn" onclick="submitForm1(); showUploadStatus()">Creează grafic predictii</button>
    <div id="status"></div>
  </form>
  <div id="graphContainer"></div>
  
</div>
`;
  } else if (option === "Optiuni avansate") {
    content.innerHTML = `<div class="dates">
  <form id="graphForm1" method="POST">
    <label for="start">Dată de început:</label>
    <input type="date" id="start" name="start_date" value="2019-07-22" />

    <label for="end">Dată de sfârșit:</label>
    <input type="date" id="end" name="end_date" value="2019-07-23" />


    <button type="button" class="submit-btn" onclick="submitFormAR(); showUploadStatus()">Creează model AR</button>
    <button type="button" class="submit-btn" onclick="submitFormMA(); showUploadStatus()">Creează model MA</button>
    <button type="button" class="submit-btn" onclick="submitFormARMA(); showUploadStatus()">Creează model ARMA</button>
    <button type="button" class="submit-btn" onclick="submitFormARIMA(); showUploadStatus()">Creează model ARIMA</button>
    <div id="status"></div>
  </form>
  <div id="graphContainer"></div>
  <div id="moreContainer"></div>`;
  }
}

function uploadCSV() {
  var fileInput = document.getElementById("csvFile");
  console.log("file taken");
  var file = fileInput.files[0];
  var formData = new FormData();
  formData.append("csv_file", file);

  fetch("/upload-csv", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      var messageContainer = document.getElementById("messageContainer");
      if (data.message === "CSV uploaded successfully") {
        messageContainer.textContent = "Fișierul a fost încărcat cu succes.";
        document.getElementById("status").textContent = "";
        generateGraph();
      } else {
        messageContainer.textContent = "Eroare la încărcarea fișierului.";
        document.getElementById("status").textContent = "";
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function createGraph() {
  var numRowsGraph = document.getElementById("numRowsGraph").value;

  if (numRowsGraph) {
    $.ajax({
      url: "/graph",
      type: "POST",
      data: { numRowsGraph: numRowsGraph },
      success: function (response) {
        var contentDiv = document.getElementById("content");
        contentDiv.innerHTML = '<img src="' + response + '">';
      },
      error: function (xhr, status, error) {
        alert("Eroare la creare graficului.");
      },
    });
  }
}

function submitForm() {
  var form = document.getElementById("graphForm");
  var formData = new FormData(form);

  fetch("/simple-graph", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data) {
        updateGraph(data.plot_data, data.Result, data.average_max_min);
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function submitForm1() {
  var form = document.getElementById("graphForm1");
  var formData = new FormData(form);

  fetch("/create-predictions", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data) {
        updateGraph1(data.plot_data, data.predicted_values, data.actual_values);
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function submitFormAR() {
  var form = document.getElementById("graphForm1");
  var formData = new FormData(form);

  fetch("/create-predictionsAR", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data1) {
        updateGraphAR(
          data.plot_data1,
          data.plot_data2,
          data.plot_data3,
          data.predicted_values,
          data.actual_values
        );
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function submitFormMA() {
  var form = document.getElementById("graphForm1");
  var formData = new FormData(form);

  fetch("/create-modelMA", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data1) {
        updateGraphMA(data.plot_data1, data.plot_data2, data.plot_data3);
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function submitFormARMA() {
  var form = document.getElementById("graphForm1");
  var formData = new FormData(form);

  fetch("/create-modelARMA", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data1) {
        updateGraphARMA(
          data.plot_data1,
          data.plot_data2,
          data.predicted_values,
          data.actual_values
        );
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function submitFormARIMA() {
  var form = document.getElementById("graphForm1");
  var formData = new FormData(form);

  fetch("/create-modelARIMA", {
    method: "POST",
    body: formData,
  })
    .then(function (response) {
      return response.json();
    })
    .then(function (data) {
      if (data.plot_data1) {
        updateGraphARIMA(
          data.plot_data1,
          data.plot_data2,
          data.predicted_values,
          data.actual_values
        );
      }
    })
    .catch(function (error) {
      console.log(error);
    });
}

function updateGraph(plot_data, Result, average_max_min) {
  var graphContainer = document.getElementById("graphContainer");

  graphContainer.innerHTML = `
    <img src="data:image/png;base64,${plot_data}" alt="Graph">
    
  <div id="more-info1">

  <table id="min_max_table">
  <tr>
    <th>Durata valori glicemice&gt;maxim</th>
    <th>Durata valori glicemice normale</th>
    <th>Durata valori glicemice&lt;minim</th>
    <th>Media valorilor glicemice</th>
    <th>Glicemie maximă</th>
    <th>Glicemie minimă</th>
  </tr>
  <tr>
    <td>${Result[0]}(${average_max_min[5]}%)</td>
    <td>${Result[1]}(${average_max_min[4]}%)</td>
    <td>${Result[2]}(${average_max_min[3]}%)</td>
    <td>${average_max_min[0]} mg/dl</td>
    <td>${average_max_min[1]} mg/dl</td>
    <td>${average_max_min[2]} mg/dl</td>
  </tr>
</table>
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function updateGraph1(plot_data, predicted_values, actual_values) {
  var graphContainer = document.getElementById("graphContainer");
  var predictedValuesHTML = "";

  for (var i = 0; i < predicted_values.length; i++) {
    let difference = predicted_values[i] - actual_values[i];
    predictedValuesHTML += "<tr>";
    predictedValuesHTML += "<td>" + predicted_values[i] + "</td>";
    predictedValuesHTML += "<td>" + actual_values[i] + "</td>";
    predictedValuesHTML += "<td>" + difference + "</td>";
    predictedValuesHTML += "</tr>";
  }

  graphContainer.innerHTML = `
    <img src="data:image/png;base64,${plot_data}" alt="Graph">
    
    <div id="more-info1">
    

  <table>
  <tr>
  <th>Valoarea prezisă(mg/dl)</th>
  <th>Valoarea reală(mg/dl)</th>
  <th>Valoarea prezisă-Valoarea reală(mg/dl)</th>
  ${predictedValuesHTML}
  </tr>
  </table>
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function updateGraphAR(
  plot_data1,
  plot_data2,
  plot_data3,
  predicted_values,
  actual_values
) {
  var graphContainer = document.getElementById("graphContainer");
  var moreContainer = document.getElementById("moreContainer");
  var predictedValuesHTML = "";

  for (var i = 0; i < predicted_values.length; i++) {
    let difference = predicted_values[i] - actual_values[i];
    predictedValuesHTML += "<tr>";
    predictedValuesHTML += "<td>" + predicted_values[i] + "</td>";
    predictedValuesHTML += "<td>" + actual_values[i] + "</td>";
    predictedValuesHTML += "<td>" + difference + "</td>";
    predictedValuesHTML += "</tr>";
  }

  graphContainer.innerHTML = `
    <h2>Grafic funcție autocorelație parțială(PACF)</h2>
    <img class="acf_plot" src="data:image/png;base64,${plot_data1}" alt="Graph">
    <h2>Grafic model autoregresiv</h2>
    <img  src="data:image/png;base64,${plot_data2}" alt="Graph">
    <h2>Grafic predicții model autoregresiv</h2>
  
    <img src="data:image/png;base64,${plot_data3}" alt="Graph">

    <table>
  <tr>
  <th>Valoarea prezisă(mg/dl)</th>
  <th>Valoarea reală(mg/dl)</th>
  <th>Valoarea prezisă-Valoarea reală(mg/dl)</th>
  ${predictedValuesHTML}
  </tr>
  </table>
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function updateGraphMA(plot_data1, plot_data2) {
  graphContainer.innerHTML = `
    <h2>Grafic funcție autocorelație (ACF)</h2>
    <img class="acf_plot" src="data:image/png;base64,${plot_data1}" alt="Graph">
    <h2>Grafic model MA</h2>
    <img src="data:image/png;base64,${plot_data2}" alt="Graph">
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function updateGraphARMA(
  plot_data1,
  plot_data2,
  predicted_values,
  actual_values
) {
  var graphContainer = document.getElementById("graphContainer");

  var predictedValuesHTML = "";
  for (var i = 0; i < predicted_values.length; i++) {
    let difference = predicted_values[i] - actual_values[i];
    predictedValuesHTML += "<tr>";
    predictedValuesHTML += "<td>" + predicted_values[i] + "</td>";
    predictedValuesHTML += "<td>" + actual_values[i] + "</td>";
    predictedValuesHTML += "<td>" + difference + "</td>";
    predictedValuesHTML += "</tr>";
  }
  graphContainer.innerHTML = `
    <h2>Grafic model ARMA </h2>
    <img src="data:image/png;base64,${plot_data1}" alt="Graph">
    <h2>Grafic predicții model ARMA</h2>
    <img src="data:image/png;base64,${plot_data2}" alt="Graph">
    <table>
  <tr>
  <th>Valoarea prezisă(mg/dl)</th>
  <th>Valoarea reală(mg/dl)</th>
  <th>Valoarea prezisă-Valoarea reală(mg/dl)</th>
  ${predictedValuesHTML}
  </tr>
  </table>
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function updateGraphARIMA(
  plot_data1,
  plot_data2,
  predicted_values,
  actual_values
) {
  var graphContainer = document.getElementById("graphContainer");

  var predictedValuesHTML = "";
  for (var i = 0; i < predicted_values.length; i++) {
    let difference = predicted_values[i] - actual_values[i];
    predictedValuesHTML += "<tr>";
    predictedValuesHTML += "<td>" + predicted_values[i] + "</td>";
    predictedValuesHTML += "<td>" + actual_values[i] + "</td>";
    predictedValuesHTML += "<td>" + difference + "</td>";
    predictedValuesHTML += "</tr>";
  }
  graphContainer.innerHTML = `
    <h2>Grafic model ARIMA </h2>
    <img src="data:image/png;base64,${plot_data1}" alt="Graph">
    <h2>Grafic predicții model ARIMA</h2>
    <img src="data:image/png;base64,${plot_data2}" alt="Graph">
    <table>
  <tr>
  <th>Valoarea prezisă(mg/dl)</th>
  <th>Valoarea reală(mg/dl)</th>
  <th>Valoarea prezisă-Valoarea reală(mg/dl)</th>
  ${predictedValuesHTML}
  </tr>
  </table>
  </div>
  `;
  document.getElementById("status").textContent = "";
}

function generateGraph(plotData) {
  // Remove any existing image element
  var existingImage = document.getElementById("graphImage");
  if (existingImage) {
    existingImage.remove();
  }

  // Create a new image element
  var imageElement = document.createElement("img");
  imageElement.id = "graphImage";
  imageElement.src = "data:image/png;base64," + plotData;
  imageElement.alt = "Graph";

  // Attach a click event listener to the image element
  imageElement.addEventListener("click", function () {
    // Open the image in a new tab
    window.open(this.src, "_blank");
  });

  // Get the container element for the graph
  var graphContainer = document.getElementById("graphContainer");

  // Append the image to the container
  graphContainer.appendChild(imageElement);
}
