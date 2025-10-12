import chroma from 'chroma-js';
import { FC, useCallback, useEffect, useState } from 'react';
import { Tab, Tabs } from 'react-bootstrap';
import { FaCloudDownloadAlt } from 'react-icons/fa';
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
} from '../core/api';
import { useAppContext } from '../core/context';

export const BertopicPage: FC = () => {
  const { projectName } = useParams();
  const {
    appContext: { currentProject, isComputing },
  } = useAppContext();
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

  return (
    // <ProjectPageLayout projectName={projectName} currentAction="explore">
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <Tabs id="panel" className="mt-3">
            <Tab eventKey="existing" title="Existing Bertopic">
              <div className="explanations">
                Compute a Bertopic on the train dataset to identify the main topics.
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
              <h4 className="subsection">Existing Bertopic</h4>
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
                    {parameters && JSON.stringify(parameters, null, 2)}
                  </details>
                </div>
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
              <div className="explanations">Using UMAP and HDBScan</div>
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
