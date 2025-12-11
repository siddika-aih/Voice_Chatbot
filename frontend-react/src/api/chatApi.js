import axiosInstance from './axiosInstance';


export  const chat = async() => {
    try {

        const response = await axiosInstance.get('/chat/');
        return response.data
        
    } catch (error) {
        console.error('error in chat api--', error);
        return error
    }
}