import React, { useState } from 'react'
import DashboardHomePage from '../Components/DashboardHome'
import NavBar from '../Components/NavBar'
import { motion } from 'framer-motion'

const Dashboard = () => {
  const [togg,setTogg]=useState(false);
  const toggnav = () => {
    setTogg((prevTogg) => !prevTogg);
  };
  return (
    <div>
        <div className='section-20' >
        {togg && <NavBar />}
        </div>
        <div className='section-80'>
            <DashboardHomePage nclose1={toggnav}/>
        </div>
    </div>
  )
}

export default Dashboard