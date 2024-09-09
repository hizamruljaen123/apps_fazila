// Inisialisasi dashboard setelah DOM siap
document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    const map = L.map('map').setView([5.0517222, 97.3078233], 9);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
    }).addTo(map);

    // Initialize year select and button for map display
    const yearSelect = document.getElementById('yearSelect');
    const yearTabs = document.getElementById('yearTabs');
    const tabContent = document.getElementById('tabContent');

    // Store API data globally after fetching once
    let floodData = [];

    // Fetch data from API once
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            floodData = data; // Store the fetched data globally
            populateYearSelect(); // Populate year options
            populateYearTabs([2023, 2024]); // Initialize nav tabs for each year
        })
        .catch(error => console.error('Error fetching data:', error));

    // Populate year select options
    function populateYearSelect() {
        const years = [2023, 2024];
        years.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });
    }

    // Handle "Tampilkan Peta" button click
    document.getElementById('showMap').onclick = function() {
        const selectedYear = parseInt(yearSelect.value, 10);
        const filteredData = floodData.filter(item => item.Tahun === selectedYear);
        displayDataOnMap(filteredData);
    };

    // Display data on map
    function displayDataOnMap(data) {
        // Clear existing markers
        map.eachLayer((layer) => {
            if (!!layer.toGeoJSON) {
                map.removeLayer(layer);
            }
        });

        // Add tile layer back to the map
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);

        // Add filtered data markers to the map
        data.forEach(item => {
            let color;
            switch(item.Prediksi) {
                case 'Aman':
                    color = 'green';
                    break;
                case 'Siaga':
                    color = 'orange';
                    break;
                case 'Awas':
                    color = 'red';
                    break;
                default:
                    color = 'blue';
            }

            L.circleMarker([item.Koordinat.Latitude, item.Koordinat.Longitude], {
                radius: 8,
                fillColor: color,
                color: color,
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map).bindPopup(`<b>${item.Wilayah}</b><br>Prediksi: ${item.Prediksi}<br>Bulan: ${item.Bulan}, Tahun: ${item.Tahun}`);
        });
    }

    // Populate year tabs dynamically for tables
    function populateYearTabs(years) {
        years.forEach((year, index) => {
            // Create the tab for the year
            const tabItem = document.createElement('li');
            tabItem.className = "nav-item";
            tabItem.innerHTML = `
                <a class="nav-link ${index === 0 ? 'active' : ''}" id="tab-${year}" data-toggle="tab" href="#content-${year}" role="tab">${year}</a>
            `;
            yearTabs.appendChild(tabItem);

            // Create the content for the tab
            const tabContentItem = document.createElement('div');
            tabContentItem.className = `tab-pane fade ${index === 0 ? 'show active' : ''}`;
            tabContentItem.id = `content-${year}`;
            tabContentItem.role = "tabpanel";
            tabContent.appendChild(tabContentItem);

            // Fill the content with the table for the year
            const filteredData = floodData.filter(item => item.Tahun === year);
            displayDataInTable(filteredData, tabContentItem);
        });
    }

    // Display data in table for each year
    function displayDataInTable(data, container) {
        const table = document.createElement('table');
        table.className = 'table table-bordered';

        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Wilayah</th>
                <th>Bulan</th>
                <th>Prediksi</th>
                <th>Koordinat (Latitude, Longitude)</th>
            </tr>
        `;

        const tbody = document.createElement('tbody');
        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.Wilayah}</td>
                <td>${item.Bulan}</td>
                <td>${item.Prediksi}</td>
                <td>${item.Koordinat.Latitude}, ${item.Koordinat.Longitude}</td>
            `;
            tbody.appendChild(row);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        container.appendChild(table);
    }
    
});
function startTraining() {
    // Set the API URL
    const apiUrl = 'http://127.0.0.1:5000/train';

    // Fetch data from the training API
    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            // Parse the response and display it in the textarea
            const result = `Message: ${data.message}\n` +
                           `Train Accuracy: ${data.train_accuracy}\n` +
                           `Test Accuracy: ${data.test_accuracy}\n` +
                           `Train MSE: ${data.train_mse}\n` +
                           `Test MSE: ${data.test_mse}`;

            // Display the result in the textarea
            document.getElementById('trainResult').value = result;
        })
        .catch(error => {
            // Handle errors
            document.getElementById('trainResult').value = 'Error: ' + error.message;
        });
}

// Array to define the custom order of short month names
const shortMonthOrder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function loadDataLatih() {
    // Fetch data latih from API
    fetch('/data_train')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#dataLatihTable tbody');
            tbody.innerHTML = ''; // Clear previous rows

            // Sort data based on shortMonthOrder and Tahun
            data.sort((a, b) => {
                const monthDiff = shortMonthOrder.indexOf(a.Bulan) - shortMonthOrder.indexOf(b.Bulan);
                if (monthDiff !== 0) {
                    return monthDiff; // Sort by month
                } else {
                    return a.Tahun - b.Tahun; // If months are the same, sort by year
                }
            });

            // Insert sorted data into the table
            data.forEach(item => {
                const row = `
                    <tr>
                        <td>${item.Wilayah}</td>
                        <td>${item.Bulan}</td>
                        <td>${item.Tahun}</td>
                        <td>${item.Curah_Hujan}</td>
                        <td>${item.Suhu}</td>
                        <td>${item.Tinggi_Muka_Air}</td>
                        <td>${item.Potensi_Banjir}</td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', row);
            });
        })
        .catch(error => console.error('Error fetching data latih:', error));
}

function loadDataUji() {
    // Fetch data uji from API
    fetch('/data_test')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#dataUjiTable tbody');
            tbody.innerHTML = ''; // Clear previous rows

            // Sort data based on shortMonthOrder and Tahun
            data.sort((a, b) => {
                const monthDiff = shortMonthOrder.indexOf(a.Bulan) - shortMonthOrder.indexOf(b.Bulan);
                if (monthDiff !== 0) {
                    return monthDiff; // Sort by month
                } else {
                    return a.Tahun - b.Tahun; // If months are the same, sort by year
                }
            });

            // Insert sorted data into the table
            data.forEach(item => {
                const row = `
                    <tr>
                        <td>${item.Wilayah}</td>
                        <td>${item.Bulan}</td>
                        <td>${item.Tahun}</td>
                        <td>${item.Curah_Hujan}</td>
                        <td>${item.Suhu}</td>
                        <td>${item.Tinggi_Muka_Air}</td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', row);
            });
        })
        .catch(error => console.error('Error fetching data uji:', error));
}
