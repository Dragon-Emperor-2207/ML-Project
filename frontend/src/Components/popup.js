import React,{useState,useEffect} from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from "framer-motion"


const PopUp=props=>{
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [loggedIn, setLoggedIn] = useState(false);
    const [userAccountNo, setUserAccountNo] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        if (loggedIn && userAccountNo !== null) {
            navigate(`/login/${userAccountNo}`);
        }
      }, [loggedIn, userAccountNo, navigate]);

    function handleLogin(e) {
        e.preventDefault()
        const url = 'http://127.0.0.1:8000/login/'

        const userData = {
        username: username,
        password: password,
        };

        fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
        })
        .then(response => {
            if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
            }
            navigate(`/login`); //Stopped Here now add for the logout button
        })
    }

    return (
        <motion.div className="popup" animate={{x:100,scale:1}} initial={{scale:0}} >
            <div className="popup-inner">
                <h2>Login</h2>
                <form onSubmit={handleLogin}>
                    <label>
                        Username:
                        <input type="text" value={username} onChange={e => setUsername(e.target.value)} />
                    </label>
                    <label>
                        Password:
                        <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
                    </label>
                    <button type="submit" onClick={props.name}>Login</button>
                    <button type="button" onClick={props.hclose}>Exit</button>
                </form>
            </div>
        </motion.div>
    )
}
export default PopUp