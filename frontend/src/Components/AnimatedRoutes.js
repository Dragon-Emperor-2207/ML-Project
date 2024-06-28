import '../App.css';
import Home from '../Pages/Home';
import AboutUs from '../Pages/About_Us';
import Pricing from '../Pages/Pricing';
import Dashboard from '../Pages/Dashboard';
import FAQ from '../Pages/FAQ'
import TransactionHistory from '../Pages/TransactionHistory';
import { AnimatePresence } from 'framer-motion';

import {
  BrowserRouter as Router,
  Route,
  Routes,
  useLocation
} from 'react-router-dom'
import React from 'react'
import MainHeader from '../Components/MainHeader';

function AnimatedRoutes() {
    const location=useLocation();
  return (
    <AnimatePresence>
    <Routes location={location} key={location.pathname}>
        <Route path='/login' element = {<Dashboard />}/>
        <Route path='/login/transactions' element = {<TransactionHistory />}/>
        <Route path='/home' element = {<Home/>}/>
        <Route path='/about' element = {<AboutUs/>}/>
        <Route path='/pricing' element = {<Pricing/>}/>
        <Route path='/faq' element = {<FAQ/>}/> */
        {}
    </Routes>
    </AnimatePresence>
  );
}

export default AnimatedRoutes;
