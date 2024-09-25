document.getElementById('demo-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const statement = document.getElementById('statement').value;

  // Show the loading spinner
  showLoadingIndicator();

  // Hide all dialogs initially
  hideAllDialogs();

  // Fetch the analysis for nature, group, and bias
  const response = await fetch('/analyze_statement', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ statement })
  });

  const result = await response.json();

  // Hide the loading spinner when the result is ready
  hideLoadingIndicator();

  // Display each dialog with a 1-second delay
  setTimeout(() => {
    document.getElementById('nature').innerText = result.nature;
    document.getElementById('dialog-nature').style.display = 'block';
  }, 1000);

  setTimeout(() => {
    document.getElementById('group').innerText = result.group;
    document.getElementById('dialog-group').style.display = 'block';
  }, 2000);

  setTimeout(() => {
    document.getElementById('bias').innerText = result.bias;
    document.getElementById('dialog-bias').style.display = 'block';
  }, 3000);

  // After the analysis is shown, ask if the user wants to generate counterspeech
  setTimeout(() => {
    document.getElementById('counterspeech-confirmation').style.display = 'block';
  }, 4000);
});

// Handle counterspeech generation
document.getElementById('yes-btn').addEventListener('click', async function() {
  const statement = document.getElementById('statement').value;

  // Hide the confirmation buttons
  document.getElementById('counterspeech-confirmation').style.display = 'none';

  // Show the loading spinner for counterspeech generation
  showLoadingIndicator();

  // Fetch the counterspeech and fact from the backend
  const response = await fetch('/generate_counterspeech', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ statement })
  });

  const result = await response.json();

  // Hide the loading spinner once the result is ready
  hideLoadingIndicator();

  // Display the counterspeech
  document.getElementById('counterspeech').innerText = result.counterspeech;
  document.getElementById('dialog-counterspeech').style.display = 'block';

  // Display the fact
  document.getElementById('fact').innerText = result.fact;
  document.getElementById('dialog-fact').style.display = 'block';
});

// If the user clicks "No", simply hide the confirmation and do nothing
document.getElementById('no-btn').addEventListener('click', function() {
  document.getElementById('counterspeech-confirmation').style.display = 'none';
});

function hideAllDialogs() {
  document.getElementById('dialog-nature').style.display = 'none';
  document.getElementById('dialog-group').style.display = 'none';
  document.getElementById('dialog-bias').style.display = 'none';
  document.getElementById('dialog-counterspeech').style.display = 'none';
  document.getElementById('dialog-fact').style.display = 'none';
  document.getElementById('counterspeech-confirmation').style.display = 'none';
}

function showLoadingIndicator() {
  document.getElementById('loading-indicator').style.display = 'block';
}

function hideLoadingIndicator() {
  document.getElementById('loading-indicator').style.display = 'none';
}
