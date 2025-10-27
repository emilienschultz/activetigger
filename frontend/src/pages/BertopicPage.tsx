import chroma from 'chroma-js';
import { FC, useCallback, useEffect, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { FaCloudDownloadAlt, FaPen, FaRegStickyNote } from 'react-icons/fa';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';
import Select from 'react-select';
import { DisplayTableTopics, Row } from '../components/DisplayTableTopics';
import { BertopicForm } from '../components/forms/BertopicForm';
import { BertopicVizSigma } from '../components/ProjectionVizSigma/BertopicVizSigma';
import {
  useDeleteBertopic,
  useDownloadBertopicClusters,
  useDownloadBertopicTopics,
  useGetBertopicProjection,
  useGetBertopicTopics,
  useGetElementById,
  useAddScheme,
  useAddLabel,
} from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject, isComputing, reFetchCurrentProject },
  } = useAppContext();
  const { notify } = useNotifications();
  const addScheme = useAddScheme(projectName || 'demo');
  const { addLabel, addLabelSetLocalScheme } = useAddLabel(projectName || 'demo', 'default');
  const deleteBertopic = useDeleteBertopic(projectName || null);
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

  const exportBertopicAsAnnotation = async (projectSlug: string, topicModelName: string | null) => {
    if (topicModelName) {
      console.log('EXPORT AS SCHEME from exportBertopicAsAnnotation');
      console.log('projectName : ', projectSlug);
      const newScheme: string = 'topic-model_' + topicModelName;
      console.log('topicModelName : ', newScheme);
      let schemeCreated: boolean = false;
      // Create the scheme
      try {
        await addScheme(newScheme, 'multiclass');
        if (reFetchCurrentProject) reFetchCurrentProject();
        notify({ type: 'success', message: `Scheme ${newScheme} created` });
        schemeCreated = true;
      } catch (error) {
        notify({ type: 'error', message: error + '' });
      }
      if (schemeCreated) {
        addLabelSetLocalScheme(newScheme);
        // Add the labels
        topics?.map((row) => {
          console.log(row.Name);
          if (row.Name) {
            addLabel(row.Name);
          }
        });
      }
    }
  };

  const test = () => {
    console.log('TEST');
  };

  return (
    // <ProjectPageLayout projectName={projectName} currentAction="explore">
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <Tabs id="panel" className="mt-3">
            <Tab eventKey="existing" title="Existing Bertopic">
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
                  className="btn btn p-0"
                  onClick={() => {
                    deleteBertopic(currentBertopic);
                    setCurrentBertopic(null);
                  }}
                >
                  <MdOutlineDeleteOutline size={30} />
                </button>
              </div>
              {currentBertopic && (
                <>
                  <div>
                    <button
                      className="btn btn-primary"
                      onClick={() => exportBertopicAsAnnotation(projectName, currentBertopic)}
                    >
                      Make scheme <FaPen />
                    </button>
                    <button
                      className="btn btn-primary mx-2"
                      onClick={() => console.log('EXPORT AS CONTEXT')}
                    >
                      Make context <FaRegStickyNote />
                    </button>
                    <button className="btn btn-primary mx-2" onClick={() => test()}>
                      TEST
                    </button>
                  </div>
                  <div>
                    <button
                      className="btn btn-primary"
                      onClick={() =>
                        currentBertopic ? downloadBertopicTopics(currentBertopic) : null
                      }
                    >
                      Topics <FaCloudDownloadAlt />
                    </button>
                    <button
                      className="btn btn-primary mx-2"
                      onClick={() =>
                        currentBertopic ? downloadBertopicClusters(currentBertopic) : null
                      }
                    >
                      Clusters <FaCloudDownloadAlt />
                    </button>
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
                <div style={{ height: '500px' }}>
                  <DisplayTableTopics data={(topics as Row[]) || []} />
                </div>
              )}
            </Tab>
            <Tab eventKey="new" title="New Bertopic">
              <div className="explanations">
                Compute a Bertopic on the train dataset to identify the main topics.
                <br />
                Using UMAP and HDBScan
              </div>
              <BertopicForm
                projectSlug={projectName || null}
                availableModels={availableModels}
                isComputing={isComputing}
              />
            </Tab>
          </Tabs>
        </div>
      </div>
    </div>
    //  </ProjectPageLayout>
  );
};
