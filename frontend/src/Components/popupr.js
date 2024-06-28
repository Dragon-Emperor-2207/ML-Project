import React,{useState} from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    
    if (parts.length === 2) {
        return parts.pop().split(';').shift();
    }
}

const PopUpR=props=>{
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [password1, setPassword1] = useState('')
    const navigate = useNavigate();

    function handleRegister(e) {
        e.preventDefault()
        const url = 'http://127.0.0.1:8000/register/'

        const userData = {
        username: username,
        password1: password,
        password2: password1
        };
        const cookie = getCookie('csrftoken');
        console.log(cookie)

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': cookie // Include a function to get the CSRF token from cookies
            },
            body: JSON.stringify(userData),
        })
        .then(response => {
            if (!response.ok) {
                return response.json();
            }
            navigate('/login')
        })
        .then(data => {
            if (data.errors) {
                console.log('Validation errors:', data.errors);
            } else {
                console.log('Successful response:', data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    return (
        <motion.div className="popup" animate={{x:100,scale:1}} initial={{scale:0}}>
            <div className="popup-inner">
                <h2>Register</h2>
                <form onSubmit={handleRegister}>
                    <label>
                        Username:
                        <input type="text" value={username} onChange={e => setUsername(e.target.value)} />
                    </label>
                    <label>
                        Password:
                        <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
                    </label>
                    <label>
                        Confirm Password:
                        <input type="password1" value={password1} onChange={e => setPassword1(e.target.value)} />
                    </label>
                    <button type="submit">Register</button>
                    <button type="button" onClick={props.hclose}>Exit</button>
                </form>
            </div>
        </motion.div>
    )
}
export default PopUpR