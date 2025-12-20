import { StrictMode } from 'react'
import ReactDOM from 'react-dom/client'
import { Route, RouterProvider, createBrowserRouter, createRoutesFromElements } from 'react-router-dom'
import { createRoot } from 'react-dom/client'
import './index.css'
import Layout from './Layout.jsx' 
import Home from './Components/Home/Home.jsx'
import Solidity from './Components/Solidity/Solidity.jsx'
import ChatBot from './Components/ChatBox/ChatBot.jsx'

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path='/' element={<Layout />}>
      <Route path='' element={<Home />}></Route>
      <Route path='solidity' element={<Solidity />}></Route>
      <Route path='chatbot' element={<ChatBot />}></Route>
    </Route>
  )
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
