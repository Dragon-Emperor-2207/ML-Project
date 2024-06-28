import React from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'

const NavBar = () => {
    const navigate=useNavigate()
    const {userAccount_no} = useParams();
    const GoTransact=()=>{
      return (
      navigate(`/login/transactions/${userAccount_no}`)
      )
    }
    const GoDash=()=>{
      return (
      navigate(`/login/${userAccount_no}`)
      )
    }
    return (
    <motion.div className='navigation-bar' initial={{ x: -100 }}
    animate={{ x: 0 }}>
        <button onClick={GoDash}>
          Home
        </button>
        <button onClick={GoTransact}>
          Transaction History
        </button>
        <button>
          Friends
        </button>
        <button>
          Groups
        </button>
        <button>
          Inbox
        </button>
        <button>
          Vault
        </button>
    </motion.div>
  )
}

export default NavBar