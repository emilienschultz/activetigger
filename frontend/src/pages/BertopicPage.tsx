import chroma from 'chroma-js';
import cx from 'classnames';
import { FC, useCallback, useEffect, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { FaCloudDownloadAlt, FaPlusCircle } from 'react-icons/fa';
import { RiFileTransferLine } from 'react-icons/ri';
import { useParams } from 'react-router-dom';
import { DisplayTableTopics, Row } from '../components/DisplayTableTopics';
import { BertopicForm } from '../components/forms/BertopicForm';
import { ModelsPillDisplay } from '../components/ModelsPillDisplay';
import { BertopicVizSigma } from '../components/ProjectionVizSigma/BertopicVizSigma';
import {
  useDeleteBertopic,
  useDownloadBertopicClusters,
  useDownloadBertopicReport,
  useDownloadBertopicTopics,
  useExportTopicsToScheme,
  useGetBertopicProjection,
  useGetBertopicTopics,
  useGetElementById,
} from '../core/api';
import { useAppContext } from '../core/context';
import { sortDatesAsStrings } from '../core/utils';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject, isComputing },
  } = useAppContext();
  const deleteBertopic = useDeleteBertopic(projectName || null);
  const exportTopicsToScheme = useExportTopicsToScheme(projectName || null);
  const { downloadBertopicTopics } = useDownloadBertopicTopics(projectName || null);
  const { downloadBertopicClusters } = useDownloadBertopicClusters(projectName || null);
  const { downloadBertopicReport } = useDownloadBertopicReport(projectName || null);
  const availableBertopic = currentProject ? currentProject.bertopic.available : [];
  const [currentBertopic, setCurrentBertopic] = useState<string | null>(null);
  const { getElementById } = useGetElementById();

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
      if (id) {
        getElementById(id, 'train').then((res) => {
          setCurrentText(String(id) + ': ' + res?.text || null);
        });
      } else setCurrentText(null);
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
        <div className="d-flex my-2" style={{ zIndex: 100 }}>
          <ModelsPillDisplay
            modelNames={Object.values(availableBertopic)
              .sort((bertopicA, bertopicB) =>
                sortDatesAsStrings(bertopicA?.time, bertopicB?.time, true),
              )
              .map((model) => (model && model.name ? model.name : ''))}
            currentModelName={currentBertopic}
            setCurrentModelName={setCurrentBertopic}
            deleteModelFunction={deleteBertopic}
          >
            <button
              onClick={() => setShowComputeNewBertopic(true)}
              className={cx('model-pill ', isComputing ? 'disabled' : '')}
              id="create-new"
            >
              <FaPlusCircle size={20} /> Compute new BERTopic
            </button>
          </ModelsPillDisplay>
        </div>
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
        <Modal
          show={showComputeNewBertopic}
          onHide={() => setShowComputeNewBertopic(false)}
          size="xl"
          id="viz-modal"
        >
          <Modal.Header closeButton>
            <Modal.Title>
              Compute Bertopic on the train dataset to identify the main topics
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
            <div style={{ margin: '10px 0px' }}>
              <button
                className="btn-primary-action"
                onClick={() => exportBertopicAsAnnotation(currentBertopic)}
              >
                Convert to scheme <RiFileTransferLine size={20} />
              </button>
              <button
                className="btn-primary-action"
                id="download-topics"
                onClick={() => (currentBertopic ? downloadBertopicTopics(currentBertopic) : null)}
              >
                Export topics <FaCloudDownloadAlt size={20} />
              </button>
              {/* <Tooltip anchorSelect="#download-topics" place="top">
                Download the table above with the following columns : Topic, Count, Name,
                <br />
                Representation and Representative Docs
              </Tooltip> */}
              <button
                className="btn-primary-action"
                id="download-clusters"
                onClick={() => (currentBertopic ? downloadBertopicClusters(currentBertopic) : null)}
              >
                Export topic per text <FaCloudDownloadAlt size={20} />
              </button>
              <button
                className="btn-primary-action"
                id="download-clusters"
                onClick={() => (currentBertopic ? downloadBertopicReport(currentBertopic) : null)}
              >
                Topic model report <FaCloudDownloadAlt size={20} />
              </button>
              {/* <Tooltip anchorSelect="#download-clusters" place="top">
                Download a table linking each element to a cluster. The table contains 2
                <br />
                columns: id and cluster
              </Tooltip> */}
            </div>
            <div style={{ height: `${80 * (1 + topics.length)}px`, margin: '15px 0px' }}>
              <DisplayTableTopics data={(topics as unknown as Row[]) || []} />
            </div>
          </>
        )}
      </div>
    </div>
    //  </ProjectPageLayout>
  );
};
