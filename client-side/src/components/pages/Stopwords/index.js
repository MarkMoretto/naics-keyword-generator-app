import { useEffect, useState } from "react"
import { Formik, Field, Form} from "formik"
// import { fetchJSON } from "../../../_utils"


// We use the curried function above to create a `getJSON` function
// from it, and we can set stuff like headers, auth, etc.
// const getJSON = fetchJSON({ method: "GET" });
const baseUrl = new URL("http://127.0.0.1:8000/stopwords")

const Stopwords = () => {
    const [stopwordsUrl, setStopwordsUrl] = useState(baseUrl)

    const [stopwords, setStopwords] = useState({})

    // const getJSON = fetchJSON({ method: "GET" });


    const onChange = e => {
        if (e.target.value.trim()) {
            setStopwordsUrl(stopwordsUrl.searchParams.set(e.target.name, e.target.value))
        } else {
            setStopwordsUrl(stopwordsUrl.searchParams.delete(e.target.name))
        }
    }

    // const getStopwords = getJSON("http://127.0.0.1:8000/stopwords?sample_size=10");

    // useEffect(() => void getStopwords().then(setStopwords).catch(console.error), [])
    useEffect(()=>{
        fetch(stopwordsUrl)
        .then(response => response.json())
        .then(data => setStopwords(data))
    })

    return (
        <>
            <h2>Random English Stopwords!</h2>
            <h4>Enter the number of words to return and press Go!</h4>
            <div>
                <Formik
                    initialValues={{
                        numStopwords: 5
                    }}
                    onSubmit={async (values) => {
                        await new Promise((resp) => setTimeout(resp, 500))
                        console.log(JSON.stringify(values, null, 4))
                    }}
                >
                    <Form>
                        <label htmlFor="id_sample_size">No. of Words:</label>
                        <Field 
                            id="id_sample_size"
                            name="sample_size"
                            placeholder="5"
                            onChange={onChange}
                        />
                        <button type="submit">Go!</button>
                    </Form>
                </Formik>
            </div>
        </>
    )
}

export default Stopwords