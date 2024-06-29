import React from 'react'

const LoanPredictor = () => {
  return (
    <div className='bg-zinc-600 flex flex-col items-center pb-3 h-screen'>
      <h2 className='text-3xl font-bold underline text-slate-200'>Loan Predictor</h2>
      <div className='flex flex-row justify-between pl-2 pr-2 w-full space-x-2 max-w-[90%]'>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex-col flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            Gender
          </label>
          <select className='bg-gray-50 border border-gray-300 text-gray-900 
text-sm rounded-lg focus:ring-blue-500 
focus:border-blue-500 block w-full p-2.5 
dark:bg-gray-700 dark:border-gray-600 
dark:placeholder-gray-400 dark:text-white 
dark:focus:ring-blue-500 
dark:focus:border-blue-500 appearance-none 
text-center outline-none'>
            <option value="male">Male</option>
            <option value="female">Female</option>
          </select>
        </div>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex-col flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            No of Dependents
          </label>
          <select className='bg-gray-50 border border-gray-300 text-gray-900 
text-sm rounded-lg focus:ring-blue-500 
focus:border-blue-500 block w-full p-2.5 
dark:bg-gray-700 dark:border-gray-600 
dark:placeholder-gray-400 dark:text-white 
dark:focus:ring-blue-500 
dark:focus:border-blue-500 appearance-none 
text-center outline-none'>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3+">3+</option>
          </select>
        </div>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex-col flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            Marraige Status
          </label>
          <select className='bg-gray-50 border border-gray-300 text-gray-900 
text-sm rounded-lg focus:ring-blue-500 
focus:border-blue-500 block w-full p-2.5 
dark:bg-gray-700 dark:border-gray-600 
dark:placeholder-gray-400 dark:text-white 
dark:focus:ring-blue-500 
dark:focus:border-blue-500 appearance-none 
text-center outline-none'>
            <option value="married">Married</option>
            <option value="single">Single</option>
          </select>
        </div>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex-col flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            Property Area
          </label>
          <select className='bg-gray-50 border border-gray-300 text-gray-900 
text-sm rounded-lg focus:ring-blue-500 
focus:border-blue-500 block w-full p-2.5 
dark:bg-gray-700 dark:border-gray-600 
dark:placeholder-gray-400 dark:text-white 
dark:focus:ring-blue-500 
dark:focus:border-blue-500 appearance-none 
text-center outline-none'>
            <option value="urban">Urban</option>
            <option value="semiurban">Semiurban</option>
            <option value="rural">rural</option>
          </select>
        </div>
      </div>
      
      <div className='flex flex-row justify-between pl-2 pr-2 w-full space-x-2 max-w-[90%] mt-2'>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            Are you a Graduate?
          </label>
          <input type='checkbox' className='flex-grow m-0  ' />
        </div>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex space-y-2 flex-grow'>
          <label className='text-gray-50 text-center'>
            Are you Self-Employed?
          </label>
          <input type='checkbox' className='flex-grow m-0' />
        </div>
        <div className='bg-transparent border-cyan-500 border-2 rounded-lg p-3 flex space-y-2 flex-grow'>
          <label className='text-gray-50'>
            Do you have a credit history?
          </label>
          <input type='checkbox' className='flex-grow m-0' />
        </div>
      </div>
    </div>
  )
}

export default LoanPredictor