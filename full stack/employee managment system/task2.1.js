const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/employee_management', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
});

// Employee Schema
const employeeSchema = new mongoose.Schema({
    name: String,
    email: String,
    password: String,
    position: String,
    department: String,
});

const Employee = mongoose.model('Employee', employeeSchema);

// Register Employee
app.post('/register', async (req, res) => {
    const { name, email, password, position, department } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);
    const newEmployee = new Employee({ name, email, password: hashedPassword, position, department });
    await newEmployee.save();
    res.status(201).send('Employee registered successfully!');
});

// Login Employee
app.post