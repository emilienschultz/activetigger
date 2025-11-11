import chroma from 'chroma-js';
import { FC, useCallback, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { RiFileTransferLine } from 'react-icons/ri';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import { Tooltip } from 'react-tooltip';
import { DisplayTableTopics, Row } from '../components/DisplayTableTopics';
import { BertopicForm } from '../components/forms/BertopicForm';
import { BertopicVizSigma } from '../components/ProjectionVizSigma/BertopicVizSigma';
import {
  useDeleteBertopic,
  useDownloadBertopicClusters,
  useDownloadBertopicTopics,
  useExportTopicsToScheme,
  useGetBertopicProjection,
  useGetBertopicTopics,
  useGetElementById,
} from '../core/api';
import { useAppContext } from '../core/context';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject, isComputing },
  } = useAppContext();
  const deleteBertopic = useDeleteBertopic(projectName || null);
  const exportTopicsToScheme = useExportTopicsToScheme(projectName || null);
  const { downloadBertopicTopics } = useDownloadBertopicTopics(projectName || null);
  const { downloadBertopicClusters } = useDownloadBertopicClusters(projectName || null);
  const availableBertopic = currentProject ? currentProject.bertopic.available : [];
  const [currentBertopic, setCurrentBertopic] = useState<string | null>(null);
  const { getElementById } = useGetElementById(projectName || null, null, null);

  const { topics, parameters, reFetchTopics } = useGetBertopicTopics(
    projectName || null,
    currentBertopic,
  );
  const { projection, reFetchProjection } = useGetBertopicProjection(
    projectName || null,
    currentBertopic,
  );
  const labels = projection?.labels;
  const currentTraining = currentProject ? Object.entries(currentProject.bertopic.training) : null;
  const availableModels = currentProject ? currentProject.bertopic.models : [];
  useEffect(() => {
    reFetchTopics();
    reFetchProjection();
  }, [currentBertopic, reFetchTopics, reFetchProjection]);

  // Action if clicked
  const [currentText, setCurrentText] = useState<string | null>(null);
  const setSelectedId = useCallback(
    // For the moment, only get something if it is the trainset
    (id?: string) => {
      if (id)
        getElementById(id, 'train').then((res) =>
          setCurrentText(String(id) + ': ' + res?.text || null),
        );
      else setCurrentText(null);
    },
    [getElementById],
  );

  const uniqueLabels = projection ? [...new Set(projection.cluster as string[])] : [];
  const colormap = chroma.scale('Paired').colors(uniqueLabels.length);
  const labelColorMapping = uniqueLabels.reduce<Record<string, string>>(
    (acc, label, index: number) => {
      acc[label as string] = colormap[index];
      return acc;
    },
    {},
  );

  const exportBertopicAsAnnotation = async (topicModelName: string | null) => {
    if (topicModelName) {
      exportTopicsToScheme(topicModelName);
    }
  };

  const [showComputeNewBertopic, setShowComputeNewBertopic] = useState(false);

  return (
    // <ProjectPageLayout projectName={projectName} currentAction="explore">
    <div className="row">
      <div className="col-12">
        {currentTraining && currentTraining?.length > 0 && (
          <div className="alert alert-info m-2">
            Current computation
            <ul>
              {Object.values(currentTraining).map(([k, v]) => (
                <li key={k}>
                  User {k} : {(v as unknown as { progress: string })?.progress}
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="d-flex w-50 my-2" style={{ zIndex: 1000 }}>
          <Select
            className="flex-grow-1"
            options={Object.keys(availableBertopic).map((e) => ({ value: e, label: e }))}
            onChange={(e) => {
              if (e) setCurrentBertopic(e.value);
            }}
            value={{ value: currentBertopic, label: currentBertopic }}
            styles={{
              menu: (provided) => ({
                ...provided,
                zIndex: 1000,
              }),
            }}
          />
          <button
            className="btn p-0"
            onClick={() => {
              deleteBertopic(currentBertopic);
              setCurrentBertopic(null);
            }}
          >
            <MdOutlineDeleteOutline size={30} />
          </button>
          <button
            className="btn p-0 convertoscheme"
            onClick={() => exportBertopicAsAnnotation(currentBertopic)}
          >
            <RiFileTransferLine size={30} />
            <Tooltip anchorSelect=".convertoscheme" place="top">
              Convert the topic to a schemes
            </Tooltip>
          </button>
        </div>

        <button
          onClick={() => setShowComputeNewBertopic(true)}
          className="btn btn-primary my-2"
          disabled={isComputing}
        >
          Compute new BERTopic
        </button>
        <Modal
          show={showComputeNewBertopic}
          onHide={() => setShowComputeNewBertopic(false)}
          size="xl"
          id="viz-modal"
        >
          <Modal.Header closeButton>
            <Modal.Title>
              Compute Bertopic on the train dataset to identify the main topics.
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <BertopicForm
              projectSlug={projectName || null}
              availableModels={availableModels}
              isComputing={isComputing}
              setStatusDisplay={setShowComputeNewBertopic}
            />
          </Modal.Body>
        </Modal>
        {currentBertopic && (
          <>
            <div>
              <details>
                <summary>Parameters</summary>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-evenly',
                  }}
                >
                  <div style={{ width: '400px' }}>
                    <h6 className="subsection">General parameters</h6>
                    <table className="table-statistics">
                      <tbody>
                        <tr>
                          <td>Language</td>
                          <td>{parameters?.bertopic_params.language}</td>
                        </tr>
                        <tr>
                          <td>Embedding model</td>
                          <td>{parameters?.bertopic_params.embedding_model}</td>
                        </tr>
                        <tr>
                          <td>Number of keywords</td>
                          <td>{parameters?.bertopic_params.top_n_words}</td>
                        </tr>
                        <tr>
                          <td>Keywords n-grams</td>
                          <td>{parameters?.bertopic_params.n_gram_range}</td>
                        </tr>
                        <tr>
                          <td>Outlier reduction</td>
                          <td>
                            {parameters?.bertopic_params.outlier_reduction ? 'True' : 'False'}
                          </td>
                        </tr>
                        <tr>
                          <td>Minimum number of characters of texts</td>
                          <td>{parameters?.bertopic_params.filter_text_length}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <div style={{ width: '400px' }}>
                      <h6 className="subsection">Dimension reduction parameters (UMAP)</h6>
                      <table className="table-statistics">
                        <tbody>
                          <tr>
                            <td>Number of neighbors</td>
                            <td>{parameters?.bertopic_params.umap_n_neighbors}</td>
                          </tr>
                          <tr>
                            <td>Number of components</td>
                            <td>{parameters?.bertopic_params.umap_n_components}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <div style={{ width: '400px' }}>
                      <h6 className="subsection">Clustering parameters (HDBSCAN)</h6>
                      <table className="table-statistics">
                        <tbody>
                          <tr>
                            <td>Clusters' mininum size</td>
                            <td>{parameters?.bertopic_params.hdbscan_min_cluster_size}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </details>
            </div>
          </>
        )}
        {projection && (
          <>
            <div style={{ height: '300px' }}>
              <BertopicVizSigma
                className={`col-12 border h-100`}
                data={
                  projection as {
                    id: unknown[];
                    x: unknown[];
                    y: unknown[];
                    cluster: string[];
                  }
                }
                setSelectedId={setSelectedId}
                labelColorMapping={labelColorMapping}
                labelDescription={labels as unknown as { [key: string]: string }}
              />
            </div>
            {currentText && (
              <div
                className="col-12"
                style={{
                  height: '200px',
                  overflow: 'hidden',
                  overflowY: 'scroll',
                  backgroundColor: '#f5f5f5',
                }}
              >
                {currentText}
              </div>
            )}
          </>
        )}
        {topics && (
          <>
            <div style={{ height: `${80 * (1 + topics.length)}px`, margin: '15px 0px' }}>
              <DisplayTableTopics data={(topics as unknown as Row[]) || []} />
            </div>
            <div style={{ margin: '10px 0px' }}>
              <button
                className="btn btn-primary"
                id="download-topics"
                onClick={() => (currentBertopic ? downloadBertopicTopics(currentBertopic) : null)}
              >
                Topics <FaCloudDownloadAlt />
              </button>
              <Tooltip anchorSelect="#download-topics" place="top">
                Download the table above with the following columns : Topic, Count, Name,
                <br />
                Representation and Representative Docs
              </Tooltip>
              <button
                className="btn btn-primary mx-2"
                id="download-clusters"
                onClick={() => (currentBertopic ? downloadBertopicClusters(currentBertopic) : null)}
              >
                Topic per text <FaCloudDownloadAlt />
              </button>
              <Tooltip anchorSelect="#download-clusters" place="top">
                Download a table linking each element to a cluster. The table contains 2
                <br />
                columns: id and cluster
              </Tooltip>
            </div>
          </>
        )}
      </div>
    </div>
    //  </ProjectPageLayout>
  );
};
