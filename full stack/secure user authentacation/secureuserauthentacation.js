const express = require('express');
const bcrypt = require('bcrypt');
const session = require('express-session');
const bodyParser = require('body-parser');

const app = express();
const PORT = 3000;

// Middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(session({
    secret: 'your_secret_key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: true } // Set to true if using HTTPS
}));

// In-memory user store (for demonstration purposes)
const users = {};

// Registration endpoint
app.post('/register', async (req, res) => {
    const { username, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);
    users[username] = { password: hashedPassword };
    res.send('User  registered!');
});

// Login endpoint
app.post('/login', async (req, res) => {
    const { username, password } = req.body;
    const user = users[username];
    if (user && await bcrypt.compare(password, user.password)) {
        req.session.user = username; // Store user in session
        res.send('Login successful!');
    } else {
        res.send('Invalid credentials');
    }
});

// Logout endpoint
app.post('/logout', (req, res) => {
    req.session.destroy();
    res.send('Logged out successfully!');
});

// Start server
app.listen(PORT, () => {
    console.log("Server running on http://localhost:${PORT}");
});